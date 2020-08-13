import torch
import torch.nn as nn
from torch import optim
import nltk
from nltk.corpus import stopwords

import sys
import tqdm
import random
from rouge import Rouge
from models.model import IntraReviewGRU, DecoderGRU, HANNiNet
# from models.setup import train_test_setup
from .base import TrainerSetup
from visualization.attention_visualization import Visualization
from torchnlp.metrics import get_moses_multi_bleu

class ReviewGeneration(TrainerSetup):
    def __init__(self, device, net_type, save_dir, voc, prerocess, 
        training_epoch=100, latent_k=32, batch_size=40, hidden_size=300, clip=50,
        num_of_reviews = 5,
        intra_method='dualFC', inter_method='dualFC', 
        learning_rate=0.00001, decoder_learning_ratio = 20.0, dropout=0,
        setence_max_len=50):
        
        super(ReviewGeneration, self).__init__(device, net_type, save_dir, voc, prerocess, training_epoch, latent_k, batch_size, hidden_size, clip, num_of_reviews, intra_method, inter_method, learning_rate, dropout, setence_max_len)

        # Default word tokens
        self.PAD_token = 0  # Used for padding short sentences
        self.SOS_token = 1  # Start-of-sentence token
        self.EOS_token = 2  # End-of-sentence token   

        self.decoder_learning_ratio = decoder_learning_ratio
        self.teacher_forcing_ratio = 1.0
        pass    

    def set_object(self, userObj, itemObj):
        self.userObj = userObj
        self.itemObj = itemObj
        pass

    def set_testing_set(self, test_on_train_data='Y'):
        if test_on_train_data == 'Y':
            self.test_on_train_data = True
        elif test_on_train_data == 'N':
            self.test_on_train_data = False

    def set_training_review_rating(self, training_review_rating):
        self.training_review_rating = training_review_rating
        pass

    def set_testing_review_rating(self, testing_review_rating):
        self.testing_review_rating = testing_review_rating
        pass

    def set_label_sentences(self, label_sentences):
        r"""
        Setup label_sentences for GRM
        """
        self.label_sentences = label_sentences
        pass

    def set_testing_batches(self, testing_batches, testing_external_memorys, testing_batch_labels, testing_asins, testing_reviewerIDs, testing_label_sentences):
        self.testing_batches = testing_batches
        self.testing_external_memorys = testing_external_memorys
        self.testing_batch_labels = testing_batch_labels
        self.testing_asins = testing_asins
        self.testing_reviewerIDs = testing_reviewerIDs
        self.testing_label_sentences = testing_label_sentences
        pass

    def set_tune_option(self, use_pretrain_item_net=False, tuning_iNet=True):
        self.use_pretrain_item_net = use_pretrain_item_net
        self.tuning_iNet = tuning_iNet
        pass    

    def _load_pretrain_item_net(self, pretrain_model_path, _ep):
        """Using pretrain hann to initial GRM"""
        
        # Initialize IntraGRU models
        IntraGRU = list()

        for idx in range(self.num_of_reviews):
            intra_model_path = '{}/IntraGRU_idx{}_epoch{}'.format(pretrain_model_path, idx, _ep)
            model = torch.load(intra_model_path)
            IntraGRU.append(model)

        # Loading InterGRU
        inter_model_path = '{}/InterGRU_epoch{}'.format(pretrain_model_path, _ep)
        InterGRU = torch.load(inter_model_path)

        return IntraGRU, InterGRU

    def _train_iteration_grm(self, iIntraGru, iInterGru, DecoderModel, iIntraGru_Opt, iInterGru_Opt, Decoder_Opt,
        concat_rating=False, _use_coverage=False, _freeze_param=False):
        """ Training each iteraction"""

        def _get_onehot_rating(r):
            _encode_rating = self._rating_to_onehot(r)
            _encode_rating = torch.tensor(_encode_rating).to(self.device)
            return _encode_rating.unsqueeze(0)


        # Initialize this epoch loss
        hann_epoch_loss = 0
        decoder_epoch_loss = 0

        for batch_ctr in tqdm.tqdm(range(len(self.training_batches[0]))): # amount of batches
            # Training each iteraction
            for idx in range(len(self.training_batch_labels)):
                
                # If turning iNet
                if(self.tuning_iNet):
                    iInterGru_Opt.zero_grad()   # initialize inter.gru opt
                    for reviews_ctr in range(len(self.training_batches)):
                        iIntraGru_Opt[reviews_ctr].zero_grad()  # # initialize intra.gru opt

                Decoder_Opt.zero_grad()

                # Get candidate user&item batches
                item = torch.tensor(self.candidate_items[idx][batch_ctr]).to(self.device)
                user = torch.tensor(self.candidate_users[idx][batch_ctr]).to(self.device)                    

                # Forward pass through intra gru
                for reviews_ctr in range(len(self.training_batches)): # iter. through reviews

                    word_batchs, lengths, ratings = self.training_batches[reviews_ctr][batch_ctr]
                    word_batchs = word_batchs.to(self.device)
                    lengths = lengths.to(self.device)

                    s_j, s_h, intra_attn = iIntraGru[reviews_ctr](
                        word_batchs, 
                        lengths, 
                        item, 
                        user
                        )
                    s_j = s_j.unsqueeze(0)

                    # concatenate reviews' rating
                    _encode_rating = _get_onehot_rating(self.training_review_rating[reviews_ctr][batch_ctr]) if concat_rating else None     # encode rating

                    # concatenate intra-reviews' review representation.
                    if(reviews_ctr == 0):
                        s_seqence = s_j
                        r_seqence = None          # initialize input rating
                        r_seqence = _encode_rating if concat_rating else None
                    else:
                        s_seqence = torch.cat((s_seqence, s_j) , 0)
                        r_seqence = torch.cat((r_seqence, _encode_rating) , 0) if concat_rating else None
                        pass

                # Forward pass through inter gru
                interInput_asin = None
                q_i, q_h, inter_attn_score, context_vector  = iInterGru(
                    s_seqence, 
                    interInput_asin, 
                    item, 
                    user,
                    review_rating = r_seqence
                    )
                r_bar = q_i.squeeze(1)

                # Caculate Square Error
                r_u_i = torch.tensor(self.training_batch_labels[idx][batch_ctr]).to(self.device)
                hann_loss = self._square_error((r_bar*(5-1)+1), r_u_i)

                # HANN loss of this epoch
                hann_epoch_loss += hann_loss


                """
                Runing Decoder
                """

                # Ground true sentences
                target_variable, target_len, _ = self.label_sentences[0][batch_ctr] 
                max_target_len = max(target_len)
                target_variable = target_variable.to(self.device)  

                # Create initial decoder input (start with SOS tokens for each sentence)
                decoder_input = torch.LongTensor(
                    [[self.SOS_token for _ in range(self.batch_size)]]
                    )
                decoder_input = decoder_input.to(self.device)

                # Set initial decoder hidden state to the inter_hidden's final hidden state
                criterion = nn.NLLLoss()
                decoder_loss = 0
                coverage_loss = 0

                # Initial decoder hidden             
                decoder_hidden = q_h

                # Determine if we are using teacher forcing this iteration
                use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

                # concatenate reviews' rating
                _encode_rating = _get_onehot_rating(r_u_i) if concat_rating else None     # encode rating

                # Forward batch of sequences through decoder one time step at a time
                if use_teacher_forcing:
                    for t in range(max_target_len):

                        if(t == 0 and _use_coverage):
                            # Set up initial coverage probability
                            initial_coverage_prob = torch.zeros(1, self.batch_size, self.voc.num_words)
                            initial_coverage_prob = initial_coverage_prob.to(self.device)
                            DecoderModel.set_coverage_prob(initial_coverage_prob, _use_coverage)
                        
                        # Forward pass through decoder
                        decoder_output, decoder_hidden, decoder_attn = DecoderModel(
                            decoder_input, 
                            decoder_hidden, 
                            context_vector,
                            _encode_rating = _encode_rating,
                            _user_emb = user,
                            _item_emb = item,
                        )
                        # Teacher forcing: next input is current target
                        decoder_input = target_variable[t].view(1, -1)  # get the row(word) of sentences
                        
                        # Coverage mechanism
                        if(_use_coverage):
                            _softmax_output = DecoderModel.get_softmax_output()
                            _current_prob = _softmax_output.unsqueeze(0)

                            if(t==0):
                                _previous_prob_sum = _current_prob
                            else:
                                # sum up previous probability
                                _previous_prob_sum = _previous_prob_sum + _current_prob
                                DecoderModel.set_coverage_prob(_previous_prob_sum, _use_coverage)

                            tmp_vec = torch.cat((_previous_prob_sum, _current_prob), dim = 0)
                            # extract min values
                            _coverage_mechanism_ = torch.min(tmp_vec, dim = 0).values

                            _coverage_mechanism_sum = torch.sum(_coverage_mechanism_, dim=1)
                            coverage_loss += torch.sum(_coverage_mechanism_sum, dim=0)
                            pass

                        # Calculate and accumulate loss
                        nll_loss = criterion(decoder_output, target_variable[t])
                        decoder_loss += nll_loss
                else:
                    for t in range(max_target_len):

                        if(t == 0 and _use_coverage):
                            # Set up initial coverage probability
                            initial_coverage_prob = torch.zeros(1, self.batch_size, self.voc.num_words)
                            initial_coverage_prob = initial_coverage_prob.to(self.device)
                            DecoderModel.set_coverage_prob(initial_coverage_prob, _use_coverage)
                        
                        # Forward pass through decoder
                        decoder_output, decoder_hidden, decoder_attn_weight = DecoderModel(
                            decoder_input, 
                            decoder_hidden, 
                            context_vector,
                            _encode_rating = _encode_rating,
                            _user_emb = user,
                            _item_emb = item
                        )
                        # No teacher forcing: next input is decoder's own current output
                        _, topi = decoder_output.topk(1)

                        decoder_input = torch.LongTensor([[topi[i][0] for i in range(self.batch_size)]])
                        decoder_input = decoder_input.to(self.device)
                        
                        # Coverage mechanism
                        if(_use_coverage):
                            _softmax_output = DecoderModel.get_softmax_output()
                            _current_prob = _softmax_output.unsqueeze(0)

                            if(t==0):
                                _previous_prob_sum = _current_prob
                            else:
                                # sum up previous probability
                                _previous_prob_sum = _previous_prob_sum + _current_prob
                                DecoderModel.set_coverage_prob(_previous_prob_sum, _use_coverage)

                            tmp_vec = torch.cat((_previous_prob_sum, _current_prob), dim = 0)
                            # extract min values
                            _coverage_mechanism_ = torch.min(tmp_vec, dim = 0).values

                            _coverage_mechanism_sum = torch.sum(_coverage_mechanism_, dim=1)
                            coverage_loss += torch.sum(_coverage_mechanism_sum, dim=0)
                            pass

                        # Calculate and accumulate loss
                        nll_loss = criterion(decoder_output, target_variable[t])
                        decoder_loss += nll_loss
                        pass
                    pass
                
                # Freeze.parameters
                if(_freeze_param):
                    for p in iInterGru.parameters():
                        p.requires_grad = False
                    for _model in iIntraGru:                        
                        for p in _model.parameters():
                            p.requires_grad = False

                # Jointly learning
                loss = hann_loss + 0.7 * (decoder_loss/max_target_len)

                # Perform backpropatation
                loss.backward()

                # If turning iNet
                if(self.tuning_iNet):
                    # Adjust iNet model weights
                    for reviews_ctr in range(len(self.training_batches)):
                        iIntraGru_Opt[reviews_ctr].step()
                    iInterGru_Opt.step()

                    # Clip gradients: gradients are modified in place
                    for reviews_ctr in range(len(self.training_batches)):            
                        _ = nn.utils.clip_grad_norm_(iIntraGru[reviews_ctr].parameters(), self.clip)
                    _ = nn.utils.clip_grad_norm_(iInterGru.parameters(), self.clip)                    

                # Adjust Decoder model weights
                Decoder_Opt.step()

                # decoder loss of this epoch
                decoder_epoch_loss += decoder_loss.item()/float(max_target_len)

        return hann_epoch_loss, decoder_epoch_loss

    def train_grm(self, isStoreModel = False, ep_to_store = 0, 
            WriteTrainLoss=False, store_every = 2,
            isCatItemVec = False, concat_rating = False, pretrain_wordVec = None, 
            _use_coverage = False):

        asin, reviewerID = self._get_asin_reviewer()
        # Initialize textual embeddings
        embedding = pretrain_wordVec if(pretrain_wordVec != None) else nn.Embedding(self.voc.num_words, self.hidden_size)

        # Initialize asin/reviewer embeddings
        asin_embedding = nn.Embedding(len(asin), self.hidden_size)
        reviewerID_embedding = nn.Embedding(len(reviewerID), self.hidden_size)   

        # Initialize IntraGRU models and optimizers
        IntraGRU = list()

        # Using pretain item net for multi-tasking
        if(self.use_pretrain_item_net):
            pretrain_model_path = 'HANN-Plus/data/pretrain_itembase_model'
            # _ep ='26'  # 0521 add rating version
            # _ep ='52'  # 0521 add rating version
            # _ep ='11'  # 0701 full interaction pre-train
            _ep ='12'  # 0705 _all_interaction6_item.rgm.full.turn1.8.1 PRETRAIN
            _ep ='36'  # 12800
            _ep ='4'   # 08/12
            IntraGRU, InterGRU = self._load_pretrain_item_net(pretrain_model_path, _ep)

            # Use appropriate device
            InterGRU = InterGRU.to(self.device)
            for idx in range(self.num_of_reviews):    
                IntraGRU[idx] = IntraGRU[idx].to(self.device)            
        
        else:

            # Append GRU model asc
            for idx in range(self.num_of_reviews):    
                IntraGRU.append(IntraReviewGRU(self.hidden_size, embedding, asin_embedding, reviewerID_embedding,  
                    latentK = self.latent_k, method=self.intra_method))
                # Use appropriate device
                IntraGRU[idx] = IntraGRU[idx].to(self.device)
                IntraGRU[idx].train()

            # Initialize InterGRU models
            InterGRU = HANNiNet(self.hidden_size, embedding, asin_embedding, reviewerID_embedding,
                    n_layers=1, dropout=self.dropout, latentK = self.latent_k, 
                    concat_rating= concat_rating)

            # Use appropriate device
            InterGRU = InterGRU.to(self.device)
            InterGRU.train()

        # Wheather tune item net
        if(self.tuning_iNet):
            iIntraGru_Opt = list()
            # Initialize IntraGRU optimizers groups
            intra_scheduler = list()

            for idx in range(self.num_of_reviews):    
                # Initialize optimizers
                iIntraGru_Opt.append(
                    optim.AdamW(
                        IntraGRU[idx].parameters(), 
                        lr=self.learning_rate, 
                        weight_decay=0.001
                        )
                    )            
                # Assuming optimizer has two groups.
                intra_scheduler.append(
                    optim.lr_scheduler.StepLR(
                        iIntraGru_Opt[idx], 
                        step_size=20, 
                        gamma=0.3)
                        )

            # Initialize IntraGRU optimizers    
            iInterGru_Opt = optim.AdamW(
                InterGRU.parameters(), 
                lr=self.learning_rate, 
                weight_decay=0.001
                )
            # Assuming optimizer has two groups.
            inter_scheduler = optim.lr_scheduler.StepLR(
                iInterGru_Opt, 
                step_size=10, 
                gamma=0.3
                )
        else:
            iIntraGru_Opt = None
            iInterGru_Opt = None

        # Initialize DecoderGRU models and optimizers
        DecoderModel = DecoderGRU(
            embedding, 
            self.hidden_size, 
            self.voc.num_words, 
            n_layers=1, 
            dropout=self.dropout
            )

        DecoderModel.set_user_embedding(reviewerID_embedding)
        DecoderModel.set_item_embedding(asin_embedding)

        # Use appropriate device
        DecoderModel = DecoderModel.to(self.device)
        DecoderModel.train()
        # Initialize DecoderGRU optimizers    
        Decoder_Opt = optim.AdamW(
            DecoderModel.parameters(), 
            lr=self.learning_rate * self.decoder_learning_ratio, 
            weight_decay=0.001
            )

        print('Models built and ready to go!')

        RMSE = sys.maxsize
        _flag = True
        _freeze_param = False

        # Training model
        for Epoch in range(self.training_epoch):

            # freeze_param
            if(_flag):
                if RMSE < 1.076 and Epoch>2:
                    _flag = False
                    _freeze_param = True
                    pass
                pass

            # Run a training iteration with batch
            hann_group_loss, decoder_group_loss = self._train_iteration_grm(
                IntraGRU, InterGRU, DecoderModel, 
                iIntraGru_Opt, iInterGru_Opt, Decoder_Opt,
                concat_rating = concat_rating,
                _use_coverage = _use_coverage,
                _freeze_param = _freeze_param 
                )

            # Wheather tune item net
            if(self.tuning_iNet):
                inter_scheduler.step()
                for idx in range(self.num_of_reviews):
                    intra_scheduler[idx].step()
                    
            num_of_iter = len(self.training_batches[0])*len(self.training_batch_labels)
        
            hann_loss_average = hann_group_loss/num_of_iter
            decoder_loss_average = decoder_group_loss/num_of_iter

            print('Epoch:{}\tItemNet(SE):{}\tNNL:{}\t'.format(Epoch, hann_loss_average, decoder_loss_average))

            if(Epoch % store_every == 0 and isStoreModel and Epoch >= ep_to_store):
                torch.save(InterGRU, R'{}/Model/InterGRU_epoch{}'.format(self.save_dir, Epoch))
                torch.save(DecoderModel, R'{}/Model/DecoderModel_epoch{}'.format(self.save_dir, Epoch))
                for idx__, IntraGRU__ in enumerate(IntraGRU):
                    torch.save(IntraGRU__, R'{}/Model/IntraGRU_idx{}_epoch{}'.format(self.save_dir, idx__, Epoch))
    
            if WriteTrainLoss:
                with open(R'{}/Loss/TrainingLoss.txt'.format(self.save_dir),'a') as file:
                    file.write('Epoch:{}\tItemNet(SE):{}\tNNL:{}\n'.format(Epoch, hann_loss_average, decoder_loss_average))

            """
            Evaluating BLEU
            """
            # evaluating
            RMSE, _nllloss, batch_bleu_score, average_rouge_score = self.evaluate_generation(
                IntraGRU, 
                InterGRU, 
                DecoderModel, 
                Epoch,
                concat_rating=concat_rating,
                write_insert_sql=True,
                write_origin=True,
                _use_coverage = _use_coverage,
                _write_mode = 'evaluate'
                )

            print('Epoch:{}\tMSE:{}\tNNL:{}\t'.format(Epoch, RMSE, _nllloss))
            with open(R'{}/Loss/ValidationLoss.txt'.format(self.save_dir),'a') as file:
                file.write('Epoch:{}\tRMSE:{}\tNNL:{}\n'.format(Epoch, RMSE, _nllloss))   

            for num, val in enumerate(batch_bleu_score):
                with open('{}/Bleu/Validation/blue{}.score.txt'.format(self.save_dir, (num+1)),'a') as file:
                    file.write('BLEU SCORE {}.ep.{}: {}\n'.format((num+1), Epoch, val))
                print('BLEU SCORE {}: {}'.format((num+1), val))

            with open('{}/Bleu/Validation/rouge.score.txt'.format(self.save_dir), 'a') as file:
                file.write('=============================\nEpoch:{}\n'.format(Epoch))
                for _rouge_method, _metrics in average_rouge_score.items():
                    for _key, _val in _metrics.items():
                        file.write('{}. {}: {}\n'.format(_rouge_method, _key, _val))
                        print('{}. {}: {}'.format(_rouge_method, _key, _val))
            
        pass

    def evaluate_generation(self, 
        IntraGRU, InterGRU, DecoderModel, Epoch, 
        concat_rating=False,
        write_origin=False,
        write_insert_sql=False,
        _use_coverage=False,
        _write_mode = 'evaluate',
        visulize_attn_epoch = 0
        ):
        
        EngStopWords = set(stopwords.words('english'))

        group_loss = 0
        decoder_epoch_loss = 0
        AttnVisualize = Visualization(self.save_dir, visulize_attn_epoch, self.num_of_reviews)


        rouge = Rouge()

        average_rouge_score = {
            'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
            'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
            'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
            }
        average_bleu_score = {
            'bleuScore-1': 0.0,
            'bleuScore-2': 0.0,
            'bleuScore-3': 0.0,
            'bleuScore-4': 0.0
        }

        def _get_onehot_rating(r):
            _encode_rating = self._rating_to_onehot(r)
            _encode_rating = torch.tensor(_encode_rating).to(self.device)
            return _encode_rating.unsqueeze(0)


        for batch_ctr in tqdm.tqdm(range(len(self.testing_batches[0]))): #how many batches
            for idx in range(len(self.testing_batch_labels)):
                for reviews_ctr in range(len(self.testing_batches)): # iter. through reviews
                                       
                    word_batchs, lengths, ratings = self.testing_batches[reviews_ctr][batch_ctr]
                    word_batchs = word_batchs.to(self.device)
                    lengths = lengths.to(self.device)

                    current_asins = torch.tensor(self.testing_asins[idx][batch_ctr]).to(self.device)
                    current_reviewerIDs = torch.tensor(self.testing_reviewerIDs[idx][batch_ctr]).to(self.device)

                    with torch.no_grad():
                        s_j, intra_hidden, intra_attn = IntraGRU[reviews_ctr](
                            word_batchs, 
                            lengths, 
                            current_asins, 
                            current_reviewerIDs
                            )
                        s_j = s_j.unsqueeze(0)


                    # Reviewer inf. for print.
                    _reviewer = self.testing_external_memorys[reviews_ctr][batch_ctr]
                    _reviewer = torch.tensor([val for val in _reviewer]).to(self.device)
                    _reviewer = _reviewer.unsqueeze(0)
                    _reviewer_cat = torch.cat((_reviewer_cat, _reviewer) , 0) if reviews_ctr>0 else _reviewer


                    # concatenate reviews' rating
                    _encode_rating = _get_onehot_rating(self.testing_review_rating[reviews_ctr][batch_ctr]) if concat_rating else None     # encode rating

                    # concatenate intra-reviews' review representation.
                    if(reviews_ctr == 0):
                        s_seqence = s_j
                        r_seqence = None          # initialize input rating
                        r_seqence = _encode_rating if concat_rating else None
                    else:
                        s_seqence = torch.cat((s_seqence, s_j) , 0)
                        r_seqence = torch.cat((r_seqence, _encode_rating) , 0) if concat_rating else None
                        pass                                                            

                    # Writing Intra-attention weight to .html file
                    if(_write_mode == 'attention'):

                        for index_ , candidateObj_ in enumerate(current_asins):

                            intra_attn_wts = intra_attn[:,index_].squeeze(1).tolist()
                            word_indexes = word_batchs[:,index_].tolist()
                            sentence, weights = AttnVisualize.wdIndex2sentences(word_indexes, self.voc.index2word, intra_attn_wts)
                            
                            new_weights = [float(wts/sum(weights[0])) for wts in weights[0]]

                            for w_index, word in enumerate(sentence[0].split()):
                                if(word in EngStopWords):
                                    new_weights[w_index] = new_weights[w_index]*0.001
                                if(new_weights[w_index]<0.0001):
                                    new_weights[w_index] = 0

                            AttnVisualize.createHTML(
                                sentence, 
                                [new_weights], 
                                reviews_ctr,
                                fname='{}@{}'.format( self.itemObj.index2asin[candidateObj_.item()], reviews_ctr)
                                )

                with torch.no_grad():
                    q_i, q_h, inter_attn_score, context_vector  = InterGRU(
                        s_seqence, 
                        None, 
                        current_asins, 
                        current_reviewerIDs,
                        review_rating = r_seqence
                        )
                    r_bar = q_i.squeeze(1)
                    r_bar = (r_bar*(5-1)+1)

                # Caculate Square loss of HANN 
                r_u_i = torch.tensor(self.testing_batch_labels[idx][batch_ctr]).to(self.device)
                hann_loss = self._mean_square_error(r_bar, r_u_i)
                group_loss += hann_loss

                """
                Greedy Search Strategy Decoder
                """
                # Create initial decoder input (start with SOS tokens for each sentence)
                decoder_input = torch.LongTensor([[self.SOS_token for _ in range(self.batch_size)]])
                decoder_input = decoder_input.to(self.device)    

                # # all one test
                # _all_one_point = [float(1.0) for _it in range(80)]
                # current_labels = torch.FloatTensor(_all_one_point).to(self.device)
                
                # Construct rating feature
                _encode_rating = _get_onehot_rating(r_u_i)

                # Set initial decoder hidden state to the inter_hidden's final hidden state
                decoder_hidden = q_h

                criterion = nn.NLLLoss()
                decoder_loss = 0
                
                # Ground true sentences
                target_batch = self.testing_label_sentences[0][batch_ctr]
                target_variable, target_len, _ = target_batch   
                target_variable = target_variable.to(self.device)  

                # Generate max length
                max_target_len = self.setence_max_len

                # Initialize tensors to append decoded words to
                all_tokens = torch.zeros([0], device=self.device, dtype=torch.long)
                all_scores = torch.zeros([0], device=self.device)            
                
                # Greedy search
                for t in range(max_target_len):

                    if(t == 0 and _use_coverage):
                        # Set up initial coverage probability
                        initial_coverage_prob = torch.zeros(1, self.batch_size, self.voc.num_words)
                        initial_coverage_prob = initial_coverage_prob.to(self.device)
                        DecoderModel.set_coverage_prob(initial_coverage_prob, _use_coverage)

                    decoder_output, decoder_hidden, decoder_attn_weight = DecoderModel(
                        decoder_input, 
                        decoder_hidden, 
                        context_vector,
                        _encode_rating = _encode_rating,
                        _user_emb = current_reviewerIDs,
                        _item_emb = current_asins
                    )
                    # No teacher forcing: next input is decoder's own current output
                    decoder_scores_, topi = decoder_output.topk(1)

                    decoder_input = torch.LongTensor([[topi[i][0] for i in range(self.batch_size)]])
                    decoder_input = decoder_input.to(self.device)

                    ds, di = torch.max(decoder_output, dim=1)

                    # Record token and score
                    all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
                    all_scores = torch.cat((all_scores, torch.t(decoder_scores_)), dim=0)                


                    # Coverage mechanism
                    if(_use_coverage):
                        _softmax_output = DecoderModel.get_softmax_output()
                        _current_prob = _softmax_output.unsqueeze(0)

                        if(t==0):
                            _previous_prob_sum = _current_prob
                        else:
                            # sum up previous probability
                            _previous_prob_sum = _previous_prob_sum + _current_prob
                            DecoderModel.set_coverage_prob(_previous_prob_sum, _use_coverage)
                            pass
                        pass

                    # Calculate and accumulate loss
                    nll_loss = criterion(decoder_output, target_variable[t])
                    decoder_loss += nll_loss
                    pass

                # decoder loss of this epoch
                decoder_epoch_loss += decoder_loss.item()/float(max_target_len)

                """
                Decode user review from search result.
                """
                _bleu_score = {
                    'bleuScore-1': 0.0,
                    'bleuScore-2': 0.0,
                    'bleuScore-3': 0.0,
                    'bleuScore-4': 0.0
                    }

                _rouge_score = {
                    'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                    'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                    'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
                    }

                for index_ , user_ in enumerate(current_reviewerIDs):
                    
                    asin_ = current_asins[index_]

                    current_user_tokens = all_tokens[:,index_].tolist()
                    decoded_words = [self.voc.index2word[token] for token in current_user_tokens if token != 0]

                    try:
                        product_title = self.asin2title[
                            self.itemObj.index2asin[asin_.item()]
                        ]
                    except Exception as ex:
                        product_title = 'None'
                        pass

                    # Show user attention
                    inter_attn_score_ = inter_attn_score.squeeze(2).t()
                    this_user_attn = inter_attn_score_[index_]
                    this_user_attn = [str(val.item()) for val in this_user_attn]
                    attn_text = ' ,'.join(this_user_attn)                    

                    this_asin_input_reviewer = _reviewer_cat.t()[index_]
                    input_reviewer = [self.userObj.index2reviewerID[val.item()] for val in this_asin_input_reviewer]                        

                    # Show original sentences
                    current_user_sen = target_variable[:,index_].tolist()
                    origin_sen = [self.voc.index2word[token] for token in current_user_sen if token != 0]


                    generate_text = str.format(
f"""
=========================
Userid & asin:{self.userObj.index2reviewerID[user_.item()]},{self.itemObj.index2asin[asin_.item()]}
title:{product_title}
pre. consumer:{' ,'.join(input_reviewer)}
Inter attn:{attn_text}
Predict:{r_bar[index_].item()}
Rating:{r_u_i[index_].item()}
Generate: {' '.join(decoded_words)}
Origin: {' '.join(origin_sen)}
"""
                    )


                    hypothesis = ' '.join(decoded_words)
                    reference = ' '.join(origin_sen)
                    #there may be several references

                    # BLEU Score Calculation
                    bleu_score_1_ = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(1, 0, 0, 0))
                    bleu_score_2_ = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0, 1, 0, 0))
                    bleu_score_3_ = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0, 0, 1, 0))
                    bleu_score_4_ = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0, 0, 0, 1))
                    sentence_bleu_score = [bleu_score_1_, bleu_score_2_, bleu_score_3_, bleu_score_4_]

                    for num, val in enumerate(sentence_bleu_score):
                        generate_text = (
                            generate_text + str.format('BLEU-{}: {}\n'.format((num+1), val))
                        )    
                    
                    # Caculate bleu score of n-gram
                    for _index, _gn in enumerate(_bleu_score):
                        _bleu_score[_gn] += sentence_bleu_score[_index]

                    if Epoch >3:
                        # ROUGE Score Calculation
                        try:
                            _rouge_score_current = rouge.get_scores(hypothesis, reference)[0]
                            for _rouge_method, _metrics in _rouge_score_current.items():
                                for _key, _val in _metrics.items():
                                    _rouge_score[_rouge_method][_key] += _val                         
                            pass
                        except Exception as msg:
                            pass
                    
                    # Write down sentences
                    if _write_mode =='generate':
                        if self.test_on_train_data :
                            fpath = (R'{}/GenerateSentences/on_train/'.format(self.save_dir))
                        else:
                            fpath = (R'{}/GenerateSentences/on_test/'.format(self.save_dir))

                        with open(fpath + 'sentences_ep{}.txt'.format(self.training_epoch),'a') as file:
                            file.write(generate_text)  

                        if (write_insert_sql):
                            # Write insert sql
                            sqlpath = (fpath + 'insert.sql')
                            self._write_generate_reviews_into_sqlfile(
                                sqlpath, 
                                self.userObj.index2reviewerID[user_.item()],
                                self.itemObj.index2asin[asin_.item()],
                                ' '.join(decoded_words)
                                )   
                
                # Average bleu score through reviewer
                for _index, _gn in enumerate(average_bleu_score):
                    average_bleu_score[_gn] += (_bleu_score[_gn]/len(current_reviewerIDs))

                if Epoch >3:
                    # Average rouge score through reviewer
                    for _rouge_method, _metrics in _rouge_score.items():
                        for _key, _val in _metrics.items():
                            average_rouge_score[_rouge_method][_key] += (_val/len(current_reviewerIDs))


        num_of_iter = len(self.testing_batches[0])*len(self.testing_batch_labels)
        
        RMSE = group_loss/num_of_iter
        _nllloss = decoder_epoch_loss/num_of_iter

        batch_bleu_score = [average_bleu_score[_gn]/num_of_iter for _gn in average_bleu_score]
        if Epoch >3:
            for _rouge_method, _metrics in average_rouge_score.items():
                for _key, _val in _metrics.items():
                    average_rouge_score[_rouge_method][_key] = _val/num_of_iter


        return RMSE, _nllloss, batch_bleu_score, average_rouge_score


    def calculate_word_frequency(self):
        r""" 
        Calculate word frequency for coverage mechanism
        """
        _word_term_freq = list()
        _total_count = 0
        
        for _word, _count in self.voc.word2count.items():
            _total_count += _count

        for _word, _count in self.voc.word2count.items():
            _word_term_freq.append(
                float(_count/_total_count)
            )
        
        return _word_term_freq

    def _write_generate_reviews_into_sqlfile(self, fpath, reviewerID, asin, generative_review, table='#table'):
        """
        Store the generative result into sql format file.
        """
        sql = (
            """
            INSERT INTO {} 
            (`reviewerID`, `asin`, `generative_review`) VALUES 
            ('{}', '{}', '{}');
            """.format(table, reviewerID, asin, generative_review)
        )

        with open(fpath,'a') as file:
            file.write(sql) 
        pass

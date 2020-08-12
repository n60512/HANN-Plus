import torch
import torch.nn as nn
from torch import optim

import tqdm
import random
from models.model import IntraReviewGRU, SubNetwork, PredictionLayer
from models._wtensorboard import _Tensorboard
from models.setup import train_test_setup

import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

class RatingRegresstion(train_test_setup):
    def __init__(self, device, net_type, save_dir, voc, prerocess, 
        training_epoch=100, latent_k=32, batch_size=40, hidden_size=300, clip=50,
        num_of_reviews = 5, 
        intra_method='dualFC', inter_method='dualFC', 
        learning_rate=0.00001, dropout=0,
        setence_max_len=50):

        super(RatingRegresstion, self).__init__(device, net_type, save_dir, voc, prerocess, training_epoch, latent_k, batch_size, hidden_size, clip, num_of_reviews, intra_method, inter_method, learning_rate, dropout, setence_max_len)
        
        self._tesorboard = _Tensorboard(self.save_dir + '/tensorboard')
        self.use_sparsity_review = False
        pass

    def set_candidate_obj(self, userObj, itemObj):
        self.itemObj = itemObj
        self.userObj = userObj
        pass

    def set_training_net_rating(self, iNet_train_rating, uNet_train_rating):
        self.iNet_train_rating = iNet_train_rating
        self.uNet_train_rating = uNet_train_rating
        pass

    def set_testing_net_rating(self, iNet_test_rating, uNet_test_rating):
        self.iNet_test_rating = iNet_test_rating
        self.uNet_test_rating = uNet_test_rating
        pass

    def set_can2sparsity(self, can2sparsity):
        self.can2sparsity = can2sparsity
        self.use_sparsity_review = True
        pass

    def set_testing_batches(self, testing_batches, testing_batch_labels, testing_asins, testing_reviewerIDs):
        self.testing_batches = testing_batches
        self.testing_batch_labels = testing_batch_labels
        self.testing_asins = testing_asins
        self.testing_reviewerIDs = testing_reviewerIDs
        pass
    
    def set_testing_correspond_batches(self, testing_correspond_batches):
        self.testing_correspond_batches = testing_correspond_batches
        pass

    def set_uNet_train(self, correspond_batches):
        self.correspond_batches = correspond_batches
        pass

    def set_uNet_num_of_reviews(self, num_of_reviews):
        self.correspond_num_of_reviews = num_of_reviews
        pass

    def set_sparsity(self, num_of_review):
        self._sparsity_review = num_of_review

    def set_ran_sparsity(self, _ran_sparsity=False, _reviews_be_chosen=None):
        self._ran_sparsity = _ran_sparsity
        self._reviews_be_chosen = _reviews_be_chosen
        pass        

    def _train_iteration(self, iIntraGru, iInterGru, iIntraGru_Opt, iInterGru_Opt,
        uIntraGru, uInterGru, uIntraGru_Opt, uInterGru_Opt, PredLayer, predLayer_Opt,
        concat_rating = True
        ):
        r""" 
        Training each iteraction

        Args:
        iIntraGru, iInterGru, iIntraGru_Opt, iInterGru_Opt,
        uIntraGru, uInterGru, uIntraGru_Opt, uInterGru_Opt, PredLayer, predLayer_Opt
        concat_rating: set for concat. rating.

        Returns:
        epoch_loss
        """
        # Initialize this epoch loss
        epoch_loss = 0

        def _get_onehot_rating(r):
            _encode_rating = self._rating_to_onehot(r)
            _encode_rating = torch.tensor(_encode_rating).to(self.device)
            return _encode_rating.unsqueeze(0)

        for batch_ctr in tqdm.tqdm(range(len(self.training_batches[0]))): # amount of batches
            # Run multiple label for training 
            for idx in range(len(self.training_batch_labels)):
                
                # Initialize optimizer
                iInterGru_Opt.zero_grad()
                uInterGru_Opt.zero_grad()
                predLayer_Opt.zero_grad()

                """
                Forward pass through item-net
                """
                for reviews_ctr in range(len(self.training_batches)): # iter. through reviews
                    
                    # Initialize optimizer
                    iIntraGru_Opt[reviews_ctr].zero_grad()

                    # Put input data into cuda
                    word_batchs, lengths, ratings = self.training_batches[reviews_ctr][batch_ctr]
                    word_batchs = word_batchs.to(self.device)
                    lengths = lengths.to(self.device)
                    item = torch.tensor(self.candidate_items[idx][batch_ctr]).to(self.device)
                    user = torch.tensor(self.candidate_users[idx][batch_ctr]).to(self.device)

                    s_j, s_h, intra_attn = iIntraGru[reviews_ctr](
                        word_batchs, 
                        lengths, 
                        item, 
                        user
                        )
                    s_j = s_j.unsqueeze(0)

                    # concatenate reviews' rating
                    _encode_rating = _get_onehot_rating(self.iNet_train_rating[reviews_ctr][batch_ctr]) if concat_rating else None     # encode rating

                    # concatenate intra-reviews' review representation.
                    if(reviews_ctr == 0):
                        s_seqence = s_j
                        r_seqence = None          # initialize input rating
                        r_seqence = _encode_rating if concat_rating else None
                    else:
                        s_seqence = torch.cat((s_seqence, s_j) , 0)
                        r_seqence = torch.cat((r_seqence, _encode_rating) , 0) if concat_rating else None
                        pass

                # Forward pass through inter-review gru
                q_i, q_h, inter_attn  = iInterGru(
                    s_seqence, 
                    item, 
                    user,
                    review_rating = r_seqence
                    )

                """
                Forward pass through user-net
                """
                for reviews_ctr in range(self.correspond_num_of_reviews): # iter. through reviews
                    
                    # Initialize optimizer
                    uIntraGru_Opt[reviews_ctr].zero_grad()
                    word_batchs, lengths, ratings = self.correspond_batches[reviews_ctr][batch_ctr]
                    word_batchs = word_batchs.to(self.device)
                    lengths = lengths.to(self.device)
                    u_item = torch.tensor(self.candidate_items[idx][batch_ctr]).to(self.device)
                    u_user = torch.tensor(self.candidate_users[idx][batch_ctr]).to(self.device)

                    s_j, s_h, intra_attn_score = uIntraGru[reviews_ctr](
                        word_batchs, 
                        lengths, 
                        u_item, 
                        u_user
                        )                    
                    s_j = s_j.unsqueeze(0)

                    # concatenate reviews' rating
                    _encode_rating = _get_onehot_rating(self.uNet_train_rating[reviews_ctr][batch_ctr]) if concat_rating else None     # encode rating
       
                    if(reviews_ctr == 0):
                        s_seqence = s_j
                        r_seqence = None          # initialize input rating
                        r_seqence = _encode_rating if concat_rating else None
                    else:
                        s_seqence = torch.cat((s_seqence, s_j) , 0)
                        r_seqence = torch.cat((r_seqence, _encode_rating) , 0) if concat_rating else None
                        pass

                # Forward pass through inter-review gru
                q_u, q_h, inter_attn  = uInterGru(
                    s_seqence, 
                    item, 
                    user,
                    review_rating = r_seqence
                    )
                q_u = q_u.squeeze(1)

                # Prediction layer
                r_bar = PredLayer(
                    q_i, 
                    q_u, 
                    item, 
                    user
                    )
                r_bar = r_bar.squeeze(1)
                 
                # Caculate Square Error
                r_u_i = torch.tensor(self.training_batch_labels[idx][batch_ctr]).to(self.device)    # grond truth
                loss = self._square_error((r_bar*(5-1)+1), r_u_i)
                
                # Perform backpropatation
                loss.backward()

                # Clip gradients: gradients are modified in place
                for reviews_ctr in range(len(self.training_batches)):            
                    _ = nn.utils.clip_grad_norm_(iIntraGru[reviews_ctr].parameters(), self.clip)
                _ = nn.utils.clip_grad_norm_(iInterGru.parameters(), self.clip)

                # Adjust model weights
                for reviews_ctr in range(len(self.training_batches)):
                    iIntraGru_Opt[reviews_ctr].step()
                for reviews_ctr in range(self.correspond_num_of_reviews):
                    uIntraGru_Opt[reviews_ctr].step()

                iInterGru_Opt.step()
                uInterGru_Opt.step()
                predLayer_Opt.step()

                epoch_loss += loss

        return epoch_loss

    def _evaluate(self, iIntraGru, iInterGru, uIntraGru, uInterGru, PredLayer,
        concat_rating=False, isWriteAttn=False, candidateObj=None, 
        visulize_attn_epoch=0):
        r"""
        Evaluation method of HANN-Plus

        Args:
        iIntraGru, iInterGru, uIntraGru, uInterGru, PredLayer: Used model.
        concat_rating: set for concat. rating.
        isWriteAttn: set for concat. rating.
        candidateObj: Writting the product name for attention file .

        Returns:
        RMSE, Accuracy, cnf_matrix
        """       

        group_loss = 0
        _accuracy = 0
        
        true_label = list()
        predict_label = list()

        def _get_onehot_rating(r):
            _encode_rating = self._rating_to_onehot(r)
            _encode_rating = torch.tensor(_encode_rating).to(self.device)
            return _encode_rating.unsqueeze(0)        

        for batch_ctr in range(len(self.testing_batches[0])): #how many batches
            for idx in range(len(self.testing_batch_labels)):
                for reviews_ctr in range(len(self.testing_batches)): #loop review 1 to 5
                    
                    word_batchs, lengths, ratings = self.testing_batches[reviews_ctr][batch_ctr]
                    word_batchs = word_batchs.to(self.device)
                    lengths = lengths.to(self.device)

                    item = torch.tensor(self.testing_asins[idx][batch_ctr]).to(self.device)
                    user = torch.tensor(self.testing_reviewerIDs[idx][batch_ctr]).to(self.device)

                    with torch.no_grad():
                        s_j, s_h, intra_attn = iIntraGru[reviews_ctr](
                            word_batchs,
                            lengths, 
                            item, 
                            user
                            )
                        s_j = s_j.unsqueeze(0)

                        # special case  ## Set sparsity  0706 saveral
                        if( False and self._ran_sparsity and (reviews_ctr >= 4)):
                            s_j = torch.zeros((1, self.batch_size, 300) , device=self.device)
                            pass

                        # concatenate reviews' rating
                        _encode_rating = _get_onehot_rating(self.iNet_test_rating[reviews_ctr][batch_ctr]) if concat_rating else None     # encode rating

                        # concatenate intra-reviews' review representation.
                        if(reviews_ctr == 0):
                            s_seqence = s_j
                            r_seqence = None          # initialize input rating
                            r_seqence = _encode_rating if concat_rating else None
                        else:
                            s_seqence = torch.cat((s_seqence, s_j) , 0)
                            r_seqence = torch.cat((r_seqence, _encode_rating) , 0) if concat_rating else None
                            pass                
                                
                with torch.no_grad():
                    q_i, q_h, inter_attn  = iInterGru(
                        s_seqence, 
                        item, 
                        user,
                        review_rating = r_seqence
                        )
                    q_i = q_i.squeeze(1)

                """
                Forward pass through user-net
                """
                for reviews_ctr in range(self.correspond_num_of_reviews): # iter. through reviews

                    word_batchs, lengths, ratings = self.testing_correspond_batches[reviews_ctr][batch_ctr]
                    word_batchs = word_batchs.to(self.device)
                    lengths = lengths.to(self.device)

                    correspond_current_asins = torch.tensor(self.testing_asins[idx][batch_ctr]).to(self.device)
                    correspond_current_reviewerIDs = torch.tensor(self.testing_reviewerIDs[idx][batch_ctr]).to(self.device)
                    
                    with torch.no_grad():
                        s_j, s_h, intra_attn_scor = uIntraGru[reviews_ctr](
                            word_batchs, 
                            lengths, 
                            correspond_current_asins, 
                            correspond_current_reviewerIDs
                            )
                        s_j = s_j.unsqueeze(0)

                        # concatenate reviews' rating
                        _encode_rating = _get_onehot_rating(self.uNet_test_rating[reviews_ctr][batch_ctr]) if concat_rating else None     # encode rating
        
                        if(reviews_ctr == 0):
                            s_seqence = s_j
                            r_seqence = None          # initialize input rating
                            r_seqence = _encode_rating if concat_rating else None
                        else:
                            s_seqence = torch.cat((s_seqence, s_j) , 0)
                            r_seqence = torch.cat((r_seqence, _encode_rating) , 0) if concat_rating else None
                            pass

                with torch.no_grad():
                    # Forward pass through inter-review gru
                    q_u, q_h, inter_attn  = uInterGru(
                        s_seqence, 
                        item, 
                        user,
                        review_rating = r_seqence
                        )
                    q_u = q_u.squeeze(1)                    

                    # Input prediction layer
                    r_bar = PredLayer(
                        q_i, 
                        q_u, 
                        item, 
                        user
                        )
                    r_bar = r_bar.squeeze(1)

                # Caculate loss 
                r_u_i = torch.tensor(self.testing_batch_labels[idx][batch_ctr]).to(self.device)
                loss = self._mean_square_error(
                    (r_bar*(5-1)+1),
                    r_u_i
                )

                group_loss += loss
                
                # Calculate accuracy
                _count = 0
                predict_rating = (r_bar*(5-1)+1).round()
                for _key, _val in enumerate(r_u_i):
                    if(predict_rating[_key] == r_u_i[_key]):
                        _count += 1
                        
                    true_label.append(int(r_u_i[_key]))
                    predict_label.append(int(predict_rating[_key]))
                    pass

                _accuracy += float(_count/len(r_u_i))

        num_of_iter = len(self.testing_batches[0])*len(self.testing_batch_labels)

        # Calculate confusion matrix
        cnf_matrix = confusion_matrix(true_label, predict_label)

        RMSE = group_loss/num_of_iter
        Accuracy = _accuracy/num_of_iter

        return RMSE, Accuracy, cnf_matrix

    def _train(self, concat_rating = True, isStoreModel = False, store_every = 2, epoch_to_store = 0, pretrain_word_embedding = None):
        
        asin, reviewerID = self._get_asin_reviewer()
        # Initialize textual embeddings
        if(pretrain_word_embedding != None):
            embedding = pretrain_word_embedding
        else:
            embedding = nn.Embedding(self.voc.num_words, self.hidden_size)

        # Initialize asin/reviewer embeddings
        asin_embedding = nn.Embedding(len(asin), self.hidden_size)
        reviewerID_embedding = nn.Embedding(len(reviewerID), self.hidden_size)   

        #---------------------------------------- Base net model construction start ------------------------------------------#
        # Initialize iIntraGru models and optimizers
        iIntraGru = list()
        iIntraGru_Opt = list()

        # Initialize iIntraGru optimizers groups
        iIntra_scheduler = list()

        # Append GRU model asc
        for idx in range(self.num_of_reviews):    
            iIntraGru.append(
                IntraReviewGRU(
                    self.hidden_size, 
                    embedding, asin_embedding, 
                    reviewerID_embedding,  
                    latentK = self.latent_k, 
                    method=self.intra_method
                    )
                )
            # Use appropriate device
            iIntraGru[idx] = iIntraGru[idx].to(self.device)
            iIntraGru[idx].train()

            # Initialize optimizers
            iIntraGru_Opt.append(
                optim.AdamW(
                    iIntraGru[idx].parameters(), 
                    lr=self.learning_rate, 
                    weight_decay=0.001
                    )
                )
            # Assuming optimizer has two groups.
            iIntra_scheduler.append(
                optim.lr_scheduler.StepLR(
                    iIntraGru_Opt[idx], 
                    step_size=20, 
                    gamma=0.3
                    )
                )

        # Initialize iInterGru models
        iInterGru = SubNetwork(
            self.hidden_size, 
            embedding, 
            asin_embedding, 
            reviewerID_embedding,
            n_layers = 1, 
            dropout = self.dropout, 
            latentK = self.latent_k, 
            concat_rating = concat_rating,
            netType = self.net_type, 
            method = self.inter_method
            )

        # Use appropriate device
        iInterGru = iInterGru.to(self.device)
        iInterGru.train()

        # Initialize iIntraGru optimizers    
        iInterGru_Opt = optim.AdamW(
            iInterGru.parameters(), 
            lr=self.learning_rate, 
            weight_decay=0.001
            )

        # Assuming optimizer has two groups.
        iInter_scheduler = optim.lr_scheduler.StepLR(
            iInterGru_Opt, 
            step_size=10, 
            gamma=0.3
            )
        #---------------------------------------- Base net model construction complete. ------------------------------------------#
        #---------------------------------------- Correspond net model construction start ------------------------------------------#
        # Initialize iIntraGru models and optimizers
        uIntraGru = list()
        uIntraGru_Opt = list()

        # Initialize iIntraGru optimizers groups
        uIntra_scheduler = list()

        # Append GRU model asc
        for idx in range(self.correspond_num_of_reviews):    
            uIntraGru.append(
                IntraReviewGRU(
                    self.hidden_size, 
                    embedding, 
                    asin_embedding, 
                    reviewerID_embedding,  
                    latentK = self.latent_k, 
                    method=self.intra_method
                )
            )

            # Use appropriate device
            uIntraGru[idx] = uIntraGru[idx].to(self.device)
            uIntraGru[idx].train()

            # Initialize optimizers
            uIntraGru_Opt.append(
                optim.AdamW(
                    uIntraGru[idx].parameters(), 
                    lr=self.learning_rate, 
                    weight_decay=0.001
                    )
                )
            
            # Assuming optimizer has two groups.
            uIntra_scheduler.append(
                optim.lr_scheduler.StepLR(
                    uIntraGru_Opt[idx], 
                    step_size=20, 
                    gamma=0.3
                    )
                )
        
        # Initialize iInterGru models
        uInterGru = SubNetwork(
            self.hidden_size, 
            embedding, 
            asin_embedding, 
            reviewerID_embedding,
            n_layers=1, 
            dropout=self.dropout, 
            latentK = self.latent_k, 
            concat_rating = concat_rating, 
            netType = self.net_type, 
            method=self.inter_method
            )

        # Use appropriate device
        uInterGru = uInterGru.to(self.device)
        uInterGru.train()

        # Initialize iIntraGru optimizers    
        uInterGru_Opt = optim.AdamW(
            uInterGru.parameters(), 
            lr=self.learning_rate, 
            weight_decay=0.001
            )

        # Assuming optimizer has two groups.
        uInter_scheduler = optim.lr_scheduler.StepLR(
            uInterGru_Opt, 
            step_size=10, 
            gamma=0.3
            )
        #---------------------------------------- Correspond net model construction complete. ------------------------------------------#
        #---------------------------------------- Final  model construction start ------------------------------------------#

        #MultiFCMultiFC
        PredLayer = PredictionLayer(
            self.hidden_size, 
            asin_embedding, 
            reviewerID_embedding, 
            dropout=self.dropout, 
            latentK = self.latent_k
            )
        # Use appropriate device
        PredLayer = PredLayer.to(self.device)

        # Initialize iIntraGru optimizers    
        PredLayer_Opt = optim.AdamW(
            PredLayer.parameters(), 
            lr=self.learning_rate, 
            weight_decay=0.001
            )

        # Assuming optimizer has two groups.
        MFC_scheduler = optim.lr_scheduler.StepLR(
            PredLayer_Opt, 
            step_size=10, 
            gamma=0.3
            )
        #---------------------------------------- Base net model construction complete. ------------------------------------------#
        
        for Epoch in range(self.training_epoch):
            # Run a training iteration with batch
            group_loss = self._train_iteration(
                iIntraGru, iInterGru, iIntraGru_Opt, iInterGru_Opt, 
                uIntraGru, uInterGru, uIntraGru_Opt, uInterGru_Opt, 
                PredLayer, PredLayer_Opt,
                concat_rating = concat_rating
                )                

            # Adjust optimizer group
            iInter_scheduler.step()
            uInter_scheduler.step()
            MFC_scheduler.step()
            for idx in range(self.correspond_num_of_reviews):
                uIntra_scheduler[idx].step
            for idx in range(self.num_of_reviews):
                iIntra_scheduler[idx].step()

            # Caculate epoch loss
            num_of_iter = len(self.training_batches[0])*len(self.training_batch_labels)
            current_loss_average = group_loss/num_of_iter
            print('Epoch:{}\tSE:{}\t'.format(Epoch, current_loss_average))

            RMSE, Accuracy, _ = self._evaluate(
                iIntraGru, iInterGru, uIntraGru, uInterGru, PredLayer,
                concat_rating = concat_rating
                )
            print('Epoch:{}\tMSE:{}\tAccuracy:{}'.format(Epoch, RMSE, Accuracy))

            # Write confusion matrix
            plt.figure()
            self.plot_confusion_matrix(
                _, 
                classes = ['1pt', '2pt', '3pt', '4pt', '5pt'],
                normalize = not True,
                title = 'confusion matrix'
                )

            plt.savefig('{}/Loss/Confusion.Matrix/_{}.png'.format(
                self.save_dir,
                Epoch
            ))    

            if(Epoch % store_every == 0 and isStoreModel and Epoch >= epoch_to_store):
                torch.save(iInterGru, R'{}/Model/InterGRU_epoch{}'.format(self.save_dir, Epoch))
                for idx__, IntraGRU__ in enumerate(iIntraGru):
                    torch.save(IntraGRU__, R'{}/Model/IntraGRU_idx{}_epoch{}'.format(self.save_dir, idx__, Epoch))

                torch.save(uInterGru, R'{}/Model/correspond_InterGRU_epoch{}'.format(self.save_dir, Epoch))
                for idx__, correspond_IntraGRU__ in enumerate(uIntraGru):
                    torch.save(correspond_IntraGRU__, R'{}/Model/correspond_IntraGRU_idx{}_epoch{}'.format(self.save_dir, idx__, Epoch))                    

                torch.save(PredLayer, R'{}/Model/MFC_epoch{}'.format(self.save_dir, Epoch))
                        
            # Write loss
            with open(R'{}/Loss/TrainingLoss.txt'.format(self.save_dir),'a') as file:
                file.write('Epoch:{}\tSE:{}\n'.format(Epoch, current_loss_average))  

            with open(R'{}/Loss/TestingLoss.txt'.format(self.save_dir),'a') as file:
                file.write('Epoch:{}\tRMSE:{}\n'.format(Epoch, RMSE))  

            with open(R'{}/Loss/Accuracy.txt'.format(self.save_dir),'a') as file:
                file.write('Epoch:{}\tAccuracy:{}\n'.format(Epoch, Accuracy))                      
        pass

    def plot_confusion_matrix(self, cm, classes, normalize = False, title = 'Confusion matrix', cmap = plt.cm.Blues):
        r"""
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        else:
            pass

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

        pass
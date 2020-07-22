from utils import options, visulizeOutput
from utils.preprocessing import Preprocess

from utils._reviewgeneration import ReviewGeneration
from visualization.attention_visualization import Visualization

import datetime
import tqdm
import torch
import torch.nn as nn
from torch import optim
import random

from gensim.models import KeyedVectors
import numpy as np
import time

# Use cuda
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
opt = options.GatherOptions().parse()

# If use pre-train word vector , load .vec
if(opt.use_pretrain_word == 'Y'):
    print('\nLoading pre-train word vector...') 
    st = time.time()    
    filename = 'HANN-Plus/data/{}festtext_subEmb.vec'.format(opt.selectTable)
    pretrain_words = KeyedVectors.load_word2vec_format(filename, binary=False)
    print('Loading complete. [{}]'.format(time.time()-st))


def _train_test(data_preprocess):

    res, itemObj, userObj = data_preprocess.load_data(
        sqlfile=opt.sqlfile, 
        mode='train', 
        table= opt.selectTable, 
        rand_seed=opt.train_test_rand_seed
        )  # for clothing.

    # Generate voc & (User or Item) information , CANDIDATE could be USER or ITEM
    voc, CANDIDATE, candiate2index = data_preprocess.generate_candidate_voc(
        res, 
        having_interaction=opt.having_interactions, 
        net_type = opt.net_type
        )

    # pre-train words
    if(opt.use_pretrain_word == 'Y'):
        weights_matrix = data_preprocess.load_pretain_word(voc, pretrain_words)
        weights_tensor = torch.FloatTensor(weights_matrix)
        pretrain_wordVec = nn.Embedding.from_pretrained(weights_tensor).to(device)           
    else:
        pretrain_wordVec = None

    """
    Construct RGM task
    """
    review_generation = ReviewGeneration(device, opt.net_type, opt.save_dir, voc, data_preprocess, 
        training_epoch=opt.epoch, latent_k=opt.latentK, batch_size=opt.batchsize, hidden_size=opt.hidden, clip=opt.clip,
        num_of_reviews = opt.num_of_reviews, 
        intra_method=opt.intra_attn_method , inter_method=opt.inter_attn_method,
        learning_rate=opt.lr, decoder_learning_ratio=opt.decoder_learning_ratio, 
        dropout=opt.dropout,
        setence_max_len=opt.setence_max_len
        )

    if(opt.mode == "train"):

        # Generate train set && candidate
        training_batch_labels, candidate_asins, candidate_reviewerIDs, label_sen_batch = data_preprocess.get_train_set(CANDIDATE, 
            itemObj, 
            userObj, 
            voc,
            batchsize=opt.batchsize, 
            num_of_reviews=5, 
            num_of_rating=1,
            net_type=opt.net_type,
            mode='generate'
            )

        if(opt.net_type == 'user_base'):
            candidateObj = itemObj
        elif(opt.net_type == 'item_base'):
            candidateObj = userObj

        # Generate `training set batches`
        training_sentences_batches, external_memorys, training_review_rating = data_preprocess.GenerateTrainingBatches(CANDIDATE, candidateObj, voc, 
            net_type = opt.net_type, 
            num_of_reviews=opt.num_of_reviews, 
            batch_size=opt.batchsize,
            get_rating_batch = True
            )

        review_generation.set_training_batches(training_sentences_batches, external_memorys, candidate_asins, candidate_reviewerIDs, training_batch_labels)

        review_generation.set_label_sentences(label_sen_batch)
        review_generation.set_tune_option(use_pretrain_item_net=True, tuning_iNet=True)
        review_generation.set_training_review_rating(training_review_rating)

    # Generate testing batches
    if(opt.mode == "eval_mse" or opt.mode == "eval_bleu" 
        or opt.mode == "generation" or opt.mode == 'attention' or opt.mode == "train"):
        
        review_generation.set_testing_set(
            test_on_train_data = opt.test_on_traindata
            )
        
        # Chose dataset for (train / validation / test)
        if (opt.mode == 'train'):
            _sql_mode = 'validation'
            # _sql_mode = 'test'
            pass
        elif (opt.mode == 'eval_bleu' or opt.mode == "generation" or opt.mode == 'eval_mse' or opt.mode == 'attention'):
            # _sql_mode = 'test'

            if(opt.test_on_traindata == 'Y'):
                _sql_mode = 'train'
            else:
                _sql_mode = 'validation'
            pass

        # Loading testing data from database
        res, itemObj, userObj = data_preprocess.load_data(
            sqlfile=opt.sqlfile, 
            mode = _sql_mode, 
            table=opt.selectTable, 
            rand_seed=opt.train_test_rand_seed, 
            test_on_train_data=review_generation.test_on_train_data
            )  
        
        # If mode:`test` , won't generate a new voc.
        CANDIDATE, candiate2index = data_preprocess.generate_candidate_voc(res, having_interaction=opt.having_interactions, generate_voc=False, 
            net_type = opt.net_type)

        testing_batch_labels, testing_asins, testing_reviewerIDs, testing_label_sentences = data_preprocess.get_train_set(
            CANDIDATE, 
            itemObj, 
            userObj, 
            voc,
            batchsize=opt.batchsize, 
            num_of_reviews=5, 
            num_of_rating=1,
            net_type=opt.net_type,
            testing=True,
            mode='generate'            
            )

        if(opt.net_type == 'user_base'):
            candidateObj = itemObj
        elif(opt.net_type == 'item_base'):
            candidateObj = userObj

        # Generate testing batches
        testing_batches, testing_external_memorys, testing_review_rating = data_preprocess.GenerateTrainingBatches(
            CANDIDATE, 
            candidateObj, 
            voc, 
            net_type = opt.net_type,
            num_of_reviews=opt.num_of_reviews, 
            batch_size=opt.batchsize, 
            testing=True,
            get_rating_batch = True
            )

        review_generation.set_testing_batches(
            testing_batches, 
            testing_external_memorys, 
            testing_batch_labels, 
            testing_asins, 
            testing_reviewerIDs, 
            testing_label_sentences
            )
        
        review_generation.set_object(userObj, itemObj)
        review_generation.set_testing_review_rating(testing_review_rating)
    
    # Set coverage mechanism
    if(opt.use_coverage == 'Y'):
        _use_coverage = True
    else:
        _use_coverage = False

    """
    Start to train GRM 
    """
    if(opt.mode == "train"):
        review_generation.train_grm(
            isStoreModel=True, 
            WriteTrainLoss=True, 
            store_every = opt.save_model_freq, 
            isCatItemVec = False, 
            concat_rating = True,
            ep_to_store=opt.epoch_to_store,
            pretrain_wordVec=pretrain_wordVec,
            _use_coverage = _use_coverage
            )


    # Testing(chose epoch)
    if(opt.mode == "generation"):

        # Set up asin2title
        review_generation.set_asin2title(
            data_preprocess.load_asin2title(
                sqlfile='HANN-Plus/SQL/cloth_asin2title.sql'
                )
        )
        # Setup epoch being chosen
        chose_epoch = opt.epoch

        # Loading IntraGRU
        IntraGRU = list()
        for idx in range(opt.num_of_reviews):
            model = torch.load(
                R'{}/Model/IntraGRU_idx{}_epoch{}'.format(
                    opt.save_dir, idx, chose_epoch
                    )
                )
            model.eval()
            IntraGRU.append(model)

        # Loading InterGRU
        InterGRU = torch.load(R'{}/Model/InterGRU_epoch{}'.format(opt.save_dir, chose_epoch))
        InterGRU.eval()
        # Loading DecoderModel
        DecoderModel = torch.load(R'{}/Model/DecoderModel_epoch{}'.format(opt.save_dir, chose_epoch))
        DecoderModel.eval()

        # evaluating
        RMSE, _nllloss, batch_bleu_score, average_rouge_score = review_generation.evaluate_generation(
            IntraGRU, 
            InterGRU, 
            DecoderModel, 
            chose_epoch,
            concat_rating = True, 
            write_insert_sql=True,
            write_origin=True,
            _use_coverage=_use_coverage,
            _write_mode = 'generate'
            )
        
        print(batch_bleu_score)

    # Testing(chose epoch)
    if(opt.mode == "eval_bleu"):

        for Epoch in range(opt.start_epoch, opt.epoch, opt.save_model_freq):
            # Loading IntraGRU
            IntraGRU = list()
            for idx in range(opt.num_of_reviews):
                model = torch.load(R'{}/Model/IntraGRU_idx{}_epoch{}'.format(
                    opt.save_dir, 
                    idx, 
                    Epoch
                    )
                )
                # model.eval()
                IntraGRU.append(model)

            # Loading InterGRU
            InterGRU = torch.load(R'{}/Model/InterGRU_epoch{}'.format(opt.save_dir, Epoch))
            # InterGRU.eval()

            # Loading DecoderModel
            DecoderModel = torch.load(R'{}/Model/DecoderModel_epoch{}'.format(opt.save_dir, Epoch))
            # DecoderModel.eval()

            print(R'{}/Model/InterGRU_epoch{}'.format(opt.save_dir, Epoch))
            print(R'{}/Model/DecoderModel_epoch{}'.format(opt.save_dir, Epoch))
        
            # evaluating
            RMSE, _nllloss, batch_bleu_score, average_rouge_score = review_generation.evaluate_generation(
                IntraGRU, 
                InterGRU, 
                DecoderModel, 
                Epoch,
                isCatItemVec=False, 
                concat_rating = True,
                write_insert_sql = True,
                write_origin = True,
                _use_coverage = _use_coverage,
                _write_mode = 'evaluate'
                )

            print('Epoch:{}\tMSE:{}\tNNL:{}\t'.format(Epoch, RMSE, _nllloss))
            with open(R'{}/Loss/TestingLoss.txt'.format(opt.save_dir),'a') as file:
                file.write('Epoch:{}\tRMSE:{}\tNNL:{}\n'.format(Epoch, RMSE, _nllloss))   

            for num, val in enumerate(batch_bleu_score):
                with open('{}/Bleu/Test/blue{}.score.txt'.format(opt.save_dir, (num+1)),'a') as file:
                    file.write('BLEU SCORE {}.ep.{}: {}\n'.format((num+1), Epoch, val))
                print('\nBLEU SCORE {}: {}'.format((num+1), val))

            with open('{}/Bleu/Test/rouge.score.txt'.format(opt.save_dir), 'a') as file:
                file.write('=============================\nEpoch:{}\n'.format(Epoch))
                for _rouge_method, _metrics in average_rouge_score.items():
                    for _key, _val in _metrics.items():
                        file.write('{}. {}: {}\n'.format(_rouge_method, _key, _val))
                        print('{}. {}: {}'.format(_rouge_method, _key, _val))

    # Testing(chose epoch)
    if(opt.mode == "attention"):

        Epoch = opt.visulize_attn_epoch

        # Loading IntraGRU
        IntraGRU = list()
        for idx in range(opt.num_of_reviews):
            model = torch.load(R'{}/Model/IntraGRU_idx{}_epoch{}'.format(
                opt.save_dir, 
                idx, 
                Epoch
                )
            )
            model.eval()
            IntraGRU.append(model)

        # Loading InterGRU
        InterGRU = torch.load(R'{}/Model/InterGRU_epoch{}'.format(opt.save_dir, Epoch))
        InterGRU.eval()

        # Loading DecoderModel
        DecoderModel = torch.load(R'{}/Model/DecoderModel_epoch{}'.format(opt.save_dir, Epoch))
        DecoderModel.eval()
    
        # evaluating
        RMSE, _nllloss, batch_bleu_score, average_rouge_score = review_generation.evaluate_generation(
            IntraGRU, 
            InterGRU, 
            DecoderModel, 
            Epoch,
            isCatItemVec=False, 
            concat_rating = True,
            write_insert_sql = True,
            write_origin = True,
            _use_coverage = _use_coverage,
            _write_mode = 'attention',
            visulize_attn_epoch=Epoch
            )

        print('Epoch:{}\tMSE:{}\tNNL:{}\t'.format(Epoch, RMSE, _nllloss))

        for num, val in enumerate(batch_bleu_score):
            print('\nBLEU SCORE {}: {}'.format((num+1), val))

        for _rouge_method, _metrics in average_rouge_score.items():
            for _key, _val in _metrics.items():
                print('{}. {}: {}'.format(_rouge_method, _key, _val))

    pass

def run():
    data_preprocess = Preprocess(setence_max_len=opt.setence_max_len)
    _train_test(data_preprocess)
    pass

if __name__ == "__main__":
    data_preprocess = Preprocess(setence_max_len=opt.setence_max_len)
    _train_test(data_preprocess)


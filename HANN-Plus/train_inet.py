from utils import options, visulizeOutput
from utils.preprocessing import Preprocess
from utils.model import IntraReviewGRU, HANN
from utils._ratingregression_baseline import RatingRegresstion
from visualization.attention_visualization import Visualization

import datetime
import tqdm
import torch
import torch.nn as nn
from torch import optim
import random

from gensim.models import KeyedVectors
import numpy as np

# Use cuda
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
opt = options.GatherOptions().parse()

"""Loading pretrain fasttext embedding"""
if(opt.use_pretrain_word == 'Y'):
    filename = 'HANN-Plus/data/{}festtext_subEmb.vec'.format(opt.selectTable)
    pretrain_words = KeyedVectors.load_word2vec_format(filename, binary=False)

def _single_model(data_preprocess):

    res, itemObj, userObj = data_preprocess.load_data(
        sqlfile=opt.sqlfile, 
        mode='train', table= opt.selectTable, rand_seed=opt.train_test_rand_seed)  # for clothing.

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


    # Construct rating regression class
    rating_regresstion = RatingRegresstion(
        device, opt.net_type, opt.save_dir, voc, data_preprocess, 
        training_epoch=opt.epoch, latent_k=opt.latentK, 
        batch_size=opt.batchsize, hidden_size=opt.hidden, clip=opt.clip,
        num_of_reviews = opt.num_of_reviews, 
        intra_method=opt.intra_attn_method , inter_method=opt.inter_attn_method,
        learning_rate=opt.lr, dropout=opt.dropout
        )


    if(opt.mode == "train"):

        # Generate train set && candidate
        training_batch_labels, candidate_asins, candidate_reviewerIDs, _ = data_preprocess.get_train_set(
            CANDIDATE, 
            itemObj, 
            userObj, 
            voc,
            batchsize=opt.batchsize, 
            num_of_reviews=5, 
            num_of_rating=1
            )

        if(opt.net_type == 'user_base'):
            candidateObj = itemObj
        elif(opt.net_type == 'item_base'):
            candidateObj = userObj

        # Generate `training set batches`
        training_sentences_batches, external_memorys, training_review_rating = data_preprocess.GenerateTrainingBatches(
            CANDIDATE, 
            candidateObj, 
            voc, 
            net_type = opt.net_type, 
            num_of_reviews=opt.num_of_reviews, 
            batch_size=opt.batchsize,
            get_rating_batch = True
            )

        # Set training batches
        rating_regresstion.set_training_batches(
            training_sentences_batches, 
            external_memorys, 
            candidate_asins, 
            candidate_reviewerIDs, 
            training_batch_labels
            )

        rating_regresstion.set_training_review_rating(training_review_rating, None)


    """Generate testing batches"""
    if(opt.mode == "test" or opt.mode == "train"):

        # Loading testing data from database
        res, itemObj, userObj = data_preprocess.load_data(
            sqlfile=opt.sqlfile, 
            mode='validation',
            table=opt.selectTable, 
            rand_seed=opt.train_test_rand_seed, 
            test_on_train_data=False
            )

        # If mode = test, won't generate a new voc.
        CANDIDATE, candiate2index = data_preprocess.generate_candidate_voc(
            res, 
            having_interaction=opt.having_interactions, 
            generate_voc=False, 
            net_type = opt.net_type
            )

        testing_batch_labels, testing_asins, testing_reviewerIDs, _ = data_preprocess.get_train_set(
            CANDIDATE, 
            itemObj, 
            userObj, 
            voc,
            batchsize=opt.batchsize, 
            num_of_reviews=5, 
            num_of_rating=1
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
            get_rating_batch=True
            )

        rating_regresstion.set_testing_batches(
            testing_batches, 
            testing_external_memorys, 
            testing_batch_labels, 
            testing_asins, 
            testing_reviewerIDs
            )

        rating_regresstion.set_testing_review_rating(
            testing_review_rating, 
            None
        )

    # Concat. rating embedding
    if(opt.concat_review_rating == 'Y'):
        concat_rating = True
    else:
        concat_rating = False

    """
    Start training process.
    """
    if(opt.mode == "train"):
        rating_regresstion.train(
            opt.selectTable, 
            isStoreModel=True, 
            WriteTrainLoss=True, 
            store_every = opt.save_model_freq, 
            use_pretrain_item=False, 
            isCatItemVec=False, 
            concat_rating=concat_rating,
            pretrain_wordVec=pretrain_wordVec
            )    

    # Testing
    if(opt.mode == "test"):
        # Evaluation (testing data)
        for Epoch in range(0, opt.epoch, opt.save_model_freq):
            # Loading IntraGRU
            IntraGRU = list()
            for idx in range(opt.num_of_reviews):
                model = torch.load(R'{}/Model/IntraGRU_idx{}_epoch{}'.format(opt.save_dir, idx, Epoch))
                model.eval()
                IntraGRU.append(model)

            # Loading InterGRU
            InterGRU = torch.load(R'{}/Model/InterGRU_epoch{}'.format(opt.save_dir, Epoch))
            InterGRU.eval()

            RMSE = rating_regresstion._itemNet_evaluate(
                IntraGRU, 
                InterGRU, 
                isWriteAttn=False,
                isCatItemVec=False, 
                concat_rating=concat_rating
                )       
                
            print('Epoch:{}\tMSE:{}\t'.format(Epoch, RMSE))

            with open(R'{}/Loss/TestingLoss.txt'.format(opt.save_dir),'a') as file:
                file.write('Epoch:{}\tRMSE:{}\n'.format(Epoch, RMSE))    

    pass

if __name__ == "__main__":


    data_preprocess = Preprocess(setence_max_len=opt.setence_max_len)
    _single_model(data_preprocess)



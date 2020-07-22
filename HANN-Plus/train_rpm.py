from utils import options
from utils.preprocessing import Preprocess
from utils._ratingregression import RatingRegresstion

import datetime
import tqdm
import torch
import torch.nn as nn
from torch import optim
import random

from gensim.models import KeyedVectors
import numpy as np
import matplotlib.pyplot as plt

# Use cuda
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
opt = options.GatherOptions().parse()

# Loading pretrain fasttext embedding
if(opt.use_pretrain_word == 'Y'):
    filename = 'HANN-Plus/data/{}festtext_subEmb.vec'.format(opt.selectTable)
    pretrain_words = KeyedVectors.load_word2vec_format(filename, binary=False)

def _train_RPM(data_preprocess):

    """
    Generate Item-Net training data.
    """
    item_net_sql = opt.sqlfile
    base_model_net_type = 'item_base'
    correspond_model_net_type = 'user_base'

    res, itemObj, userObj = data_preprocess.load_data(
        sqlfile=item_net_sql, 
        mode='train', 
        table= opt.selectTable, 
        rand_seed=opt.train_test_rand_seed
        )

    # Generate voc & (User or Item) information , CANDIDATE could be USER or ITEM
    voc, ITEM, candiate2index = data_preprocess.generate_candidate_voc(
        res, 
        having_interaction=opt.having_interactions, 
        net_type = base_model_net_type
        )

    # pre-train words
    if(opt.use_pretrain_word == 'Y'):
        weights_matrix = data_preprocess.load_pretain_word(voc, pretrain_words)
        weights_tensor = torch.FloatTensor(weights_matrix)
        pretrain_word_vec = nn.Embedding.from_pretrained(weights_tensor).to(device)           
    else:
        pretrain_word_vec = None
    
    # Generate train set && candidate
    training_batch_labels, candidate_asins, candidate_reviewerIDs, _ = data_preprocess.get_train_set(ITEM, 
        itemObj, 
        userObj, 
        voc,
        batchsize = opt.batchsize, 
        num_of_reviews = 5, 
        num_of_rating = 1
        )

    if(base_model_net_type == 'user_base'):
        candidateObj = itemObj
    elif(base_model_net_type == 'item_base'):
        candidateObj = userObj

    # Generate `training set batches`
    training_batches, training_item_batches, iNet_rating = data_preprocess.GenerateTrainingBatches(
        ITEM, 
        candidateObj, 
        voc, 
        net_type = base_model_net_type, 
        num_of_reviews=opt.num_of_reviews, 
        batch_size=opt.batchsize,
        get_rating_batch = True
        )
    
    """
    Generate User-Net training data.
    """
    num_of_reviews_uNet = opt.num_of_correspond_reviews
    user_base_sql = R'HANN-Plus/SQL/_all_interaction6_item.candidate.user.sql'

    res, itemObj, userObj = data_preprocess.load_data(
        sqlfile=user_base_sql, 
        mode='train', 
        table= opt.selectTable, 
        rand_seed=opt.train_test_rand_seed
        )

    # Generate voc & (User or Item) information , CANDIDATE could be USER or ITEM
    USER, uid2index = data_preprocess.generate_candidate_voc(
        res, 
        having_interaction=opt.having_interactions, 
        net_type = correspond_model_net_type,
        generate_voc=False
        )

    # Export the `Consumer's history` through chosing number of `candidate`
    chosing_num_of_candidate = opt.num_of_reviews
    ITEM_CONSUMER = list()
    for _item in ITEM:
        candidate_uid = _item.this_reviewerID[chosing_num_of_candidate]
        user_index = uid2index[candidate_uid]   

        # Append the user which is chosen into consumer list
        ITEM_CONSUMER.append(USER[user_index])   

    # Generate correspond net batches
    correspond_batches, correspond_asin_batches, uNet_rating = data_preprocess.GenerateTrainingBatches(
        ITEM_CONSUMER, itemObj, voc, 
        net_type = correspond_model_net_type,
        num_of_reviews= num_of_reviews_uNet, 
        batch_size=opt.batchsize,
        testing=True,
        get_rating_batch = True
        )

    # Start to train model by `rating regression`
    rating_regresstion = RatingRegresstion(
        device, opt.net_type, opt.save_dir, voc, data_preprocess, 
        training_epoch=opt.epoch, latent_k=opt.latentK, 
        batch_size=opt.batchsize,
        hidden_size=opt.hidden, clip=opt.clip,
        num_of_reviews = opt.num_of_reviews, 
        intra_method=opt.intra_attn_method , inter_method=opt.inter_attn_method,
        learning_rate=opt.lr, dropout=opt.dropout
        )

    """
    Training setup
    """
    rating_regresstion.set_training_batches(training_batches, training_item_batches, candidate_asins, candidate_reviewerIDs, training_batch_labels)
    rating_regresstion.set_uNet_train(correspond_batches)                   # this method for hybird only
    rating_regresstion.set_uNet_num_of_reviews(num_of_reviews_uNet)           # this method for hybird only
    rating_regresstion.set_training_net_rating(iNet_rating, uNet_rating)
    rating_regresstion.set_candidate_obj(userObj, itemObj)

    # Chose dataset for (train / validation / test)
    if (opt.mode == 'train'):
        _sql_mode = 'validation'
        pass
    elif (opt.mode == 'test' or opt.mode == 'attention'):
        _sql_mode = 'test'
        _sql_mode = 'validation'
        pass

    """Creating testing batches"""
    # Loading testing data from database
    res, itemObj, userObj = data_preprocess.load_data(
        sqlfile=opt.sqlfile, 
        mode=_sql_mode, 
        table=opt.selectTable, 
        rand_seed=opt.train_test_rand_seed
        )

    # If mode = test, won't generate a new voc.
    CANDIDATE, candiate2index = data_preprocess.generate_candidate_voc(
        res, having_interaction=opt.having_interactions, generate_voc=False, 
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

    # Generate testing batches
    testing_batches, testing_external_memorys, testing_review_rating = data_preprocess.GenerateTrainingBatches(
        CANDIDATE, 
        userObj, 
        voc, 
        net_type = opt.net_type,
        num_of_reviews=opt.num_of_reviews, 
        batch_size=opt.batchsize, 
        testing=True,
        get_rating_batch = True
        )

    if(opt.hybird == 'Y'):
        if(opt.sqlfile_fill_user==''):
            user_base_sql = R'HANN-Plus/SQL/_all_interaction6_item.candidate.user.sql'
        else:
            user_base_sql = opt.sqlfile_fill_user   # select the generative table

        res, itemObj, userObj = data_preprocess.load_data(
            sqlfile = user_base_sql, 
            mode=_sql_mode, 
            table = opt.selectTable, 
            rand_seed = opt.train_test_rand_seed,
            num_of_generative=opt.num_of_generative
            )  

        # Generate USER information 
        USER, uid2index = data_preprocess.generate_candidate_voc(
            res, 
            having_interaction = opt.having_interactions, 
            net_type = 'user_base',
            generate_voc = False
            )

        # Create item consumer list
        ITEM_CONSUMER = list()
        for _item in CANDIDATE:
            candidate_uid = _item.this_reviewerID[5]
            user_index = uid2index[candidate_uid]

            ITEM_CONSUMER.append(USER[user_index])

        """Enable useing sparsity review (training set `OFF`)"""
        """Using this when testing on sparsity reviews"""
        if(opt.use_sparsity_review == 'Y'):
            # loading sparsity review
            can2sparsity = data_preprocess.load_sparsity_reviews(
                opt.sparsity_pickle
                )       
            # setup can2sparsity
            rating_regresstion.set_can2sparsity(can2sparsity)

            # Replace reviews by sparsity
            for _index, user in enumerate(ITEM_CONSUMER):
                # load target user's sparsity list
                sparsity_list = can2sparsity[user.reviewerID]
                
                # Replace reviews to null by `sparsity_list`
                for _num_of_review , _val in enumerate(sparsity_list):
                    if(_val == 0):
                        ITEM_CONSUMER[_index].sentences[_num_of_review] = ''
            
        # Generate `training correspond set batches`
        correspond_batches, correspond_asin_batches, correspond_review_rating = data_preprocess.GenerateTrainingBatches(
            ITEM_CONSUMER, 
            itemObj, 
            voc, 
            net_type = 'user_base', 
            num_of_reviews= opt.num_of_correspond_reviews, 
            batch_size=opt.batchsize,
            testing=True,
            get_rating_batch = True
            )
        
        rating_regresstion.set_testing_correspond_batches(correspond_batches)

        pass

    """Setting testing setup"""
    rating_regresstion.set_testing_batches(
        testing_batches, 
        testing_batch_labels, 
        testing_asins, 
        testing_reviewerIDs
    )

    # Set testing net rating
    rating_regresstion.set_testing_net_rating(testing_review_rating, correspond_review_rating)

    # Concat. rating embedding
    concat_rating = True if(opt.concat_review_rating == 'Y') else False

    # Set random sparsity
    _ran_sparsity = True if(opt._ran_sparsity == 'Y') else False
    _reviews_be_chosen = opt._reviews_be_chosen if(opt._ran_sparsity == 'Y') else None
    rating_regresstion.set_ran_sparsity(_ran_sparsity = _ran_sparsity, _reviews_be_chosen = _reviews_be_chosen)

    """Start training"""
    if(opt.mode == 'train'):
        rating_regresstion._train(
            concat_rating=concat_rating,
            isStoreModel=True,  
            store_every = opt.save_model_freq, 
            epoch_to_store = opt.epoch_to_store,
            pretrain_word_embedding=pretrain_word_vec
            )


if __name__ == "__main__":

    data_preprocess = Preprocess(setence_max_len=opt.setence_max_len)
    _train_RPM(data_preprocess)
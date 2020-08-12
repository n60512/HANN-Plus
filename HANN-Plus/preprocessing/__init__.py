from .preprocessing import Preprocess
from trainers._ratingregression import RatingRegresstion
from trainers._reviewgeneration import ReviewGeneration
from gensim.models import KeyedVectors

import torch
import torch.nn as nn

def rpm_preprocess(args, device):
    
    # Loading pretrain fasttext embedding
    if(args.use_pretrain_word == 'Y'):
        filename = 'HANN-Plus/data/{}festtext_subEmb.vec'.format(args.selectTable)
        pretrain_words = KeyedVectors.load_word2vec_format(filename, binary=False)

    data_preprocess = Preprocess(setence_max_len=args.setence_max_len)

    """
    Generate Item-Net training data.
    """
    item_net_sql = args.sqlfile
    base_model_net_type = 'item_base'
    correspond_model_net_type = 'user_base'

    res, itemObj, userObj = data_preprocess.load_data(
        sqlfile=item_net_sql, 
        mode= 'train', 
        table= args.selectTable, 
        rand_seed=args.train_test_rand_seed
        )

    # Generate voc & (User or Item) information , CANDIDATE could be USER or ITEM
    voc, ITEM, candiate2index = data_preprocess.generate_candidate_voc(
        res, 
        having_interaction=args.having_interactions, 
        net_type = base_model_net_type
        )

    # pre-train words
    if(args.use_pretrain_word == 'Y'):
        weights_matrix = data_preprocess.load_pretain_word(voc, pretrain_words)
        weights_tensor = torch.FloatTensor(weights_matrix)
        pretrain_word_embedd = nn.Embedding.from_pretrained(weights_tensor).to(device)           
    else:
        pretrain_word_embedd = None
    
    # Generate train set && candidate
    training_batch_labels, candidate_asins, candidate_reviewerIDs, _ = data_preprocess.get_train_set(ITEM, 
        itemObj, 
        userObj, 
        voc,
        batchsize = args.batchsize, 
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
        num_of_reviews=args.num_of_reviews, 
        batch_size=args.batchsize,
        get_rating_batch = True
        )
    
    """
    Generate User-Net training data.
    """
    num_of_reviews_uNet = args.num_of_correspond_reviews
    user_base_sql = R'HANN-Plus/SQL/_all_interaction6_item.candidate.user.sql'

    res, itemObj, userObj = data_preprocess.load_data(
        sqlfile=user_base_sql, 
        mode='train', 
        table= args.selectTable, 
        rand_seed=args.train_test_rand_seed
        )

    # Generate voc & (User or Item) information , CANDIDATE could be USER or ITEM
    USER, uid2index = data_preprocess.generate_candidate_voc(
        res, 
        having_interaction=args.having_interactions, 
        net_type = correspond_model_net_type,
        generate_voc=False
        )

    # Export the `Consumer's history` through chosing number of `candidate`
    chosing_num_of_candidate = args.num_of_reviews
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
        batch_size=args.batchsize,
        testing=True,
        get_rating_batch = True
        )


    # Start to train model by `rating regression`
    rating_regresstion = RatingRegresstion(
        device, args.net_type, args.save_dir, voc, data_preprocess, 
        training_epoch=args.epoch, latent_k=args.latentK, 
        batch_size=args.batchsize,
        hidden_size=args.hidden, clip=args.clip,
        num_of_reviews = args.num_of_reviews, 
        intra_method=args.intra_attn_method , inter_method=args.inter_attn_method,
        learning_rate=args.lr, dropout=args.dropout
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
    if (args.mode == 'train'):
        _sql_mode = 'validation'
        pass
    elif (args.mode == 'test' or args.mode == 'attention'):
        _sql_mode = 'test'
        _sql_mode = 'validation'
        pass

    """Creating testing batches"""
    # Loading testing data from database
    res, itemObj, userObj = data_preprocess.load_data(
        sqlfile=args.sqlfile, 
        mode=_sql_mode, 
        table=args.selectTable, 
        rand_seed=args.train_test_rand_seed
        )

    # If mode = test, won't generate a new voc.
    CANDIDATE, candiate2index = data_preprocess.generate_candidate_voc(
        res, having_interaction=args.having_interactions, generate_voc=False, 
        net_type = args.net_type
        )

    testing_batch_labels, testing_asins, testing_reviewerIDs, _ = data_preprocess.get_train_set(
        CANDIDATE, 
        itemObj, 
        userObj, 
        voc,
        batchsize=args.batchsize, 
        num_of_reviews=5, 
        num_of_rating=1
        )

    # Generate testing batches
    testing_batches, testing_external_memorys, testing_review_rating = data_preprocess.GenerateTrainingBatches(
        CANDIDATE, 
        userObj, 
        voc, 
        net_type = args.net_type,
        num_of_reviews=args.num_of_reviews, 
        batch_size=args.batchsize, 
        testing=True,
        get_rating_batch = True
        )

    """ Correspond net """
    if(args.sqlfile_fill_user==''):
        user_base_sql = R'HANN-Plus/SQL/_all_interaction6_item.candidate.user.sql'
    else:
        user_base_sql = args.sqlfile_fill_user   # select the generative table

    res, itemObj, userObj = data_preprocess.load_data(
        sqlfile = user_base_sql, 
        mode=_sql_mode, 
        table = args.selectTable, 
        rand_seed = args.train_test_rand_seed,
        num_of_generative=args.num_of_generative
        )  

    # Generate USER information 
    USER, uid2index = data_preprocess.generate_candidate_voc(
        res, 
        having_interaction = args.having_interactions, 
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
    if(args.use_sparsity_review == 'Y'):
        # loading sparsity review
        can2sparsity = data_preprocess.load_sparsity_reviews(
            args.sparsity_pickle
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
        num_of_reviews= args.num_of_correspond_reviews, 
        batch_size=args.batchsize,
        testing=True,
        get_rating_batch = True
        )
    
    rating_regresstion.set_testing_correspond_batches(correspond_batches)


    """Setting testing setup"""
    rating_regresstion.set_testing_batches(
        testing_batches, 
        testing_batch_labels, 
        testing_asins, 
        testing_reviewerIDs
    )

    # Set testing net rating
    rating_regresstion.set_testing_net_rating(testing_review_rating, correspond_review_rating)

    # Set random sparsity
    _ran_sparsity = True if(args._ran_sparsity == 'Y') else False
    _reviews_be_chosen = args._reviews_be_chosen if(args._ran_sparsity == 'Y') else None
    rating_regresstion.set_ran_sparsity(_ran_sparsity = _ran_sparsity, _reviews_be_chosen = _reviews_be_chosen)

    return rating_regresstion, pretrain_word_embedd

def rgm_preprocess(args, device):

    # Loading pretrain fasttext embedding
    if(args.use_pretrain_word == 'Y'):
        filename = 'HANN-Plus/data/{}festtext_subEmb.vec'.format(args.selectTable)
        pretrain_words = KeyedVectors.load_word2vec_format(filename, binary=False)    
    
    data_preprocess = Preprocess(setence_max_len=args.setence_max_len)

    res, itemObj, userObj = data_preprocess.load_data(
        sqlfile=args.sqlfile, 
        mode='train', 
        table= args.selectTable, 
        rand_seed=args.train_test_rand_seed
        )  # for clothing.

    # Generate voc & (User or Item) information , CANDIDATE could be USER or ITEM
    voc, CANDIDATE, candiate2index = data_preprocess.generate_candidate_voc(
        res, 
        having_interaction=args.having_interactions, 
        net_type = args.net_type
        )

    # pre-train words
    if(args.use_pretrain_word == 'Y'):
        weights_matrix = data_preprocess.load_pretain_word(voc, pretrain_words)
        weights_tensor = torch.FloatTensor(weights_matrix)
        pretrain_word_embedd = nn.Embedding.from_pretrained(weights_tensor).to(device)           
    else:
        pretrain_word_embedd = None

    """
    Construct RGM task
    """
    review_generation = ReviewGeneration(device, args.net_type, args.save_dir, voc, data_preprocess, 
        training_epoch=args.epoch, latent_k=args.latentK, batch_size=args.batchsize, hidden_size=args.hidden, clip=args.clip,
        num_of_reviews = args.num_of_reviews, 
        intra_method=args.intra_attn_method , inter_method=args.inter_attn_method,
        learning_rate=args.lr, decoder_learning_ratio=args.decoder_learning_ratio, 
        dropout=args.dropout,
        setence_max_len=args.setence_max_len
        )

    if(args.mode == "train"):

        # Generate train set && candidate
        training_batch_labels, candidate_asins, candidate_reviewerIDs, label_sen_batch = data_preprocess.get_train_set(CANDIDATE, 
            itemObj, 
            userObj, 
            voc,
            batchsize=args.batchsize, 
            num_of_reviews=5, 
            num_of_rating=1,
            net_type=args.net_type,
            mode='generate'
            )

        if(args.net_type == 'user_base'):
            candidateObj = itemObj
        elif(args.net_type == 'item_base'):
            candidateObj = userObj

        # Generate `training set batches`
        training_sentences_batches, external_memorys, training_review_rating = data_preprocess.GenerateTrainingBatches(CANDIDATE, candidateObj, voc, 
            net_type = args.net_type, 
            num_of_reviews=args.num_of_reviews, 
            batch_size=args.batchsize,
            get_rating_batch = True
            )

        review_generation.set_training_batches(training_sentences_batches, external_memorys, candidate_asins, candidate_reviewerIDs, training_batch_labels)

        review_generation.set_label_sentences(label_sen_batch)
        review_generation.set_tune_option(use_pretrain_item_net=True, tuning_iNet=True)
        review_generation.set_training_review_rating(training_review_rating)

    # Generate testing batches
    if(args.mode == "eval_mse" or args.mode == "eval_bleu" 
        or args.mode == "generation" or args.mode == 'attention' or args.mode == "train"):
        
        review_generation.set_testing_set(
            test_on_train_data = args.test_on_traindata
            )
        
        # Chose dataset for (train / validation / test)
        if (args.mode == 'train'):
            _sql_mode = 'validation'
            # _sql_mode = 'test'
            pass
        elif (args.mode == 'eval_bleu' or args.mode == "generation" or args.mode == 'eval_mse' or args.mode == 'attention'):
            # _sql_mode = 'test'

            if(args.test_on_traindata == 'Y'):
                _sql_mode = 'train'
            else:
                _sql_mode = 'validation'
            pass

        # Loading testing data from database
        res, itemObj, userObj = data_preprocess.load_data(
            sqlfile=args.sqlfile, 
            mode = _sql_mode, 
            table=args.selectTable, 
            rand_seed=args.train_test_rand_seed, 
            test_on_train_data=review_generation.test_on_train_data
            )  
        
        # If mode:`test` , won't generate a new voc.
        CANDIDATE, candiate2index = data_preprocess.generate_candidate_voc(res, having_interaction=args.having_interactions, generate_voc=False, 
            net_type = args.net_type)

        testing_batch_labels, testing_asins, testing_reviewerIDs, testing_label_sentences = data_preprocess.get_train_set(
            CANDIDATE, 
            itemObj, 
            userObj, 
            voc,
            batchsize=args.batchsize, 
            num_of_reviews=5, 
            num_of_rating=1,
            net_type=args.net_type,
            testing=True,
            mode='generate'            
            )

        if(args.net_type == 'user_base'):
            candidateObj = itemObj
        elif(args.net_type == 'item_base'):
            candidateObj = userObj

        # Generate testing batches
        testing_batches, testing_external_memorys, testing_review_rating = data_preprocess.GenerateTrainingBatches(
            CANDIDATE, 
            candidateObj, 
            voc, 
            net_type = args.net_type,
            num_of_reviews=args.num_of_reviews, 
            batch_size=args.batchsize, 
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
    if(args.use_coverage == 'Y'):
        _use_coverage = True
    else:
        _use_coverage = False    
    
    return review_generation, pretrain_word_embedd
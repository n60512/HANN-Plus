from trainers._ratingregression import RatingRegresstion
from preprocessing import rpm_preprocess
import torch

# Use cuda
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def _train_RPM(args):

    # Concat. rating embedding
    concat_rating = True if(args.concat_review_rating == 'Y') else False
    
    # Preprocessing
    rating_regresstion, pretrain_word_embedd = rpm_preprocess(args, device)
    
    if(args.mode == 'train'):
        rating_regresstion._train(
            concat_rating = concat_rating,
            isStoreModel = True,  
            store_every = args.save_model_freq, 
            epoch_to_store = args.epoch_to_store,
            pretrain_word_embedding = pretrain_word_embedd
            )    
    pass
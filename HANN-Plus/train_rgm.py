from trainers._reviewgeneration import ReviewGeneration
from preprocessing import rgm_preprocess
import torch

# Use cuda
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def _train_RGM(args):

    # Preprocessing
    review_generation, pretrain_word_embedd = rgm_preprocess(args, device)

    # Set coverage mechanism
    if(args.use_coverage == 'Y'):
        _use_coverage = True
    else:
        _use_coverage = False 

    if(args.mode == "train"):
        review_generation.train_grm(
            isStoreModel = True, 
            WriteTrainLoss = True, 
            store_every = args.save_model_freq, 
            isCatItemVec = False, 
            concat_rating = True,
            ep_to_store = args.epoch_to_store,
            pretrain_wordVec = pretrain_word_embedd,
            _use_coverage = _use_coverage
            )
    pass
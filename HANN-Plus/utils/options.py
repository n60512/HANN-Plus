import argparse
import json
import os
import warnings
import datetime

class GatherOptions():
    def __init__(self):
        parser = argparse.ArgumentParser(description="HANN-Plus model")

        parser.add_argument(
            "--model", 
            default = "rpm", 
            choices = ["rpm", "rgm"],
            )

        parser.add_argument(
            "--mode", 
            default="train", 
            choices=["train", "test", 'both', 'showAttn', 'generation', 'eval_bleu', 'eval_mse', 'attention'],
            help="train or test the model" 
            )
        
        current_time = datetime.datetime.now()
        parser.add_argument("--save_dir", default=("HANN-Plus/log/origin/{:%Y%m%d_%H_%M}".format(current_time)), help="path for saving model")
        parser.add_argument("--minor_path", default='', help="another save place")
        parser.add_argument("--sparsity_pickle", default='', help="")
        
        parser.add_argument('--sqlfile', default='', help="loacl sql cmd file")
        parser.add_argument('--sqlfile_fill_user', default='', help="loacl sql cmd file")
        parser.add_argument('--num_of_generative', default='', help="chose the number that wanna fill into sparsity")

        parser.add_argument("--train_test_rand_seed", type=int, default=42, help="rand seed for data seleting")        

        parser.add_argument('--save_model_freq', type=int, default=1, help="frequency of saving model")
        parser.add_argument('--epoch_to_store', type=int, default=0, help="")
        
        parser.add_argument("--setence_max_len", type=int, default=100, help="Max length of sentence")        
        parser.add_argument("--having_interactions", type=int, default=15, help="num of user interactions")        
        parser.add_argument("--epoch", type=int, default=30, help="num of eopch for training")        
        parser.add_argument("--start_epoch", type=int, default=0, help="num of eopch for start testing")

        parser.add_argument('--num_of_reviews', type=int        , default=4, help="number of every user's reviews")
        parser.add_argument('--num_of_correspond_reviews', type=int, default=4, help="number of correspond net reviews")

        parser.add_argument("--batchsize", type=int, default=40, help="input batch size")
        parser.add_argument("--num_of_rating", type=int, default=3, help="number of rating")
        parser.add_argument("--num_of_validate", type=int, default=3, help="number of validate")
        parser.add_argument("--latentK", type=int, default=32, help="latenK")
        parser.add_argument("--hidden", type=int, default=300, help="Gru hidden size")
        parser.add_argument('--lr', type=float, default=0.00005, help="initial learning rate for adam")
        parser.add_argument('--dropout', type=float, default=0, help="dropout")
        parser.add_argument('--clip', type=float, default=50.0, help="clip")
        parser.add_argument('--decoder_learning_ratio', type=float, default=1.0, help="decoder learning ratio")
        parser.add_argument('--teacher_forcing_ratio', type=float, default=1.0, help="teacher forcing ratio")

        parser.add_argument('--net_type', default='user_base', help="select net type(user or item base)")
        parser.add_argument('--selectTable', default='clothing_', help="select db table")

        parser.add_argument('--intra_attn_method', default='dualFC', help="intra attention method")
        parser.add_argument('--inter_attn_method', default='general', help="inter attention method")
        parser.add_argument('--decoder_init_method', default='concat', help="select decoder method")

        parser.add_argument("--use_pretrain_word", default="Y", choices=["Y", "N"], help="Wheather using pretrain embedding")
        parser.add_argument("--use_sparsity_review", default="N", choices=["Y", "N"], help="Wheather using sparsity reviews")       

        parser.add_argument("--_ran_sparsity", default="N", choices=["Y", "N"], help="Wheather using random sparsity reviews")
        parser.add_argument("--_reviews_be_chosen", type=int, default=None, help="")


        parser.add_argument("--use_coverage", default="N", choices=["Y", "N"], help="grm")       
        parser.add_argument("--concat_item", default="N", choices=["Y", "N"], help="")
        parser.add_argument("--concat_review_rating", default="N", choices=["Y", "N"], help="")

        parser.add_argument('--selectAttnModel', default='', help="Select model that wanna to show attn weight")
        parser.add_argument("--visulize_attn_epoch", type=int, default=0, help="No. of epoch that you like to show attention weight")
        
        parser.add_argument("--hybird", default="N", choices=["Y", "N"], help="hybird model")    

        parser.add_argument("--test_on_traindata", default="N", choices=["Y", "N"], help="Selecting testing on test/train set")

        self.parser = parser

    def parse(self, argv=None):
        if argv == None:
            opt = self.parser.parse_args(argv) # for running in jupyter notebook    
        else:
            opt = self.parser.parse_args()
        self.opt = opt
        self.config_path = os.path.join(opt.save_dir, 'opt.json')

        if (opt.mode == "test" or opt.mode == "eval_bleu" or  opt.mode == 'generation'):
            self.parser.add_argument("--model_dir", help="path to load model for test(the largest step or use --step to specify)")
            os.makedirs(opt.save_dir + "/Bleu/Test", exist_ok=True)
            
        if opt.mode == "train" or opt.mode == "both":
            os.makedirs(opt.save_dir, exist_ok=True)
            os.makedirs(opt.save_dir + "/Loss", exist_ok=True)
            os.makedirs(opt.save_dir + "/Model", exist_ok=True)
            os.makedirs(opt.save_dir + "/Bleu/Validation", exist_ok=True)
            os.makedirs(opt.save_dir + "/checkpts", exist_ok=True)
            os.makedirs(opt.save_dir + "/Loss/Confusion.Matrix", exist_ok=True)

            with open(self.config_path, 'w') as f:
                json.dump(self.opt.__dict__, f)
        
        if opt.mode == "showAttn" or opt.mode == "attention":
            os.makedirs(opt.save_dir + "/VisualizeAttn/epoch_{}".format(opt.visulize_attn_epoch), exist_ok=True)

        if opt.mode == "generation" or opt.mode == "train":
            os.makedirs(opt.save_dir + "/GenerateSentences", exist_ok=True)        
            if opt.test_on_traindata == "Y":
                os.makedirs(opt.save_dir + "/GenerateSentences/on_train", exist_ok=True)
            elif opt.test_on_traindata == "N":
                os.makedirs(opt.save_dir + "/GenerateSentences/on_test", exist_ok=True)

        return opt

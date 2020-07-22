import torch

class train_test_setup():
    def __init__(self, device, net_type, save_dir, voc, prerocess, 
        training_epoch=100, latent_k=32, batch_size=40, hidden_size=300, clip=50,
        num_of_reviews = 5, 
        intra_method='dualFC', inter_method='dualFC', 
        learning_rate=0.00001, dropout=0, setence_max_len=50):

        self.device = device
        self.net_type = net_type
        self.save_dir = save_dir
        
        self.voc = voc
        self.prerocess = prerocess              # prerocess method
        self.training_epoch = training_epoch
        self.latent_k = latent_k
        self.hidden_size = hidden_size
        self.num_of_reviews = num_of_reviews
        self.clip = clip

        self.intra_method = intra_method
        self.inter_method = inter_method
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.batch_size = batch_size
        self.setence_max_len = setence_max_len
    
        pass

    def _get_asin_reviewer(self, select_table='clothing_'):
        """Get asin and reviewerID from file"""
        asin, reviewerID = self.prerocess._read_asin_reviewer(table=select_table)
        return asin, reviewerID

    def set_training_batches(self, training_batches, external_memorys, candidate_items, candidate_users, training_batch_labels):
        self.training_batches = training_batches
        self.external_memorys = external_memorys
        self.candidate_items = candidate_items
        self.candidate_users = candidate_users
        self.training_batch_labels = training_batch_labels
        pass
    
    def set_asin2title(self, asin2title):
        self.asin2title = asin2title
        pass

    def _rating_to_onehot(self, rating, rating_dim=5, _fix_rating_experiment = -1):
        r"""
        Convert rating value into one-hot encoding.

        Args:
        rating: list of rating

        Returns:
        _encode_rating: one-hot vector of rating list
        """        

        # Initial onehot table
        onehot_table = [0 for _ in range(rating_dim)]
        rating = [int(val-1) for val in rating]
        
        _encode_rating = list()
        for val in rating:
            current_onehot = onehot_table.copy()    # copy from init.

            # This is for fixed rating experiment
            if(_fix_rating_experiment>-1):
                #fix rating
                current_onehot[_fix_rating_experiment] = 1.0               # set rating as onehot
                _encode_rating.append(current_onehot)
            else:    
                current_onehot[val] = 1.0               # set rating as onehot
                _encode_rating.append(current_onehot)

        return _encode_rating


    def _square_error(self, predict, target):
        r"""
        Calculate square error
        """
        err = predict - target
        _se = torch.mul(err, err)
        _se = torch.mean(_se, dim=0)
        return _se

    def _mean_square_error(self, predict, target):
        r"""
        Calculate mean square error
        """
        err = predict - target
        _se = torch.mul(err, err)
        _se = torch.mean(_se, dim=0)
        _se = torch.sqrt(_se)
        return _se
    
    def _accuracy(self, predict_rating, target):
        r""" 
        Calculate accuracy
        """
        _total = len(target)
        _count = 0
        true_label = list()
        predict_label = list()
        for _key, _val in enumerate(target):
            if(predict_rating[_key] == target[_key]):
                _count += 1

            true_label.append(int(target[_key]))
            predict_label.append(int(predict_rating[_key]))
        
        _acc = float(_count/_total)

        return _acc, true_label, predict_label
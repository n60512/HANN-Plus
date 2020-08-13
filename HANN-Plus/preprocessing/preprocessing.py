import torch
import re
import unicodedata
import itertools
import time
import numpy as np
import tqdm
import pickle
from utils.DBconnector import DBConnection

PAD_token = 0  # Used for padding short sentences

class Voc:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD"}
        self.num_words = 1

        self.word2count["PAD"] = 1

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1
    pass

class CandidateHistory(object):
    def __init__(self, net_type):
        self.net_type = net_type
        self.sentences = list()
        self.rating = list()
        self.this_asin = list()
        self.this_reviewerID = list()
        self.row_count = 0

    def add_data(self, sentence_, rating_, this_asin_, reviewerID_):
        pass

    def get_row_count(self):
        return self.row_count

class ReviewerHistory(CandidateHistory):
    def __init__(self, reviewerID):
        super(ReviewerHistory, self).__init__('user_base')
        self.reviewerID = reviewerID

    def add_data(self, sentence_, rating_, this_asin_, reviewerID_):
        self.sentences.append(sentence_)
        self.rating.append(rating_)
        self.this_asin.append(this_asin_)
        self.this_reviewerID.append(reviewerID_)
        self.row_count += 1

class AsinHistory(CandidateHistory):
    def __init__(self, asin):
        super(AsinHistory, self).__init__('item_base')
        self.asin = asin

    def add_data(self, sentence_, rating_, this_asin_, reviewerID_):
        self.sentences.append(sentence_)
        self.rating.append(rating_)
        self.this_asin.append(this_asin_)
        self.this_reviewerID.append(reviewerID_)
        self.row_count += 1    

class item:
    def __init__(self):
        self.asin2index = {}
        self.index2asin = {}
        self.num_asins = 0

    def addItem(self, asin):
        for id_ in asin:
            self.asin2index[id_] = self.num_asins
            self.index2asin[self.num_asins] = id_
            self.num_asins += 1

class user:
    def __init__(self):
        self.reviewerID2index = {}
        self.index2reviewerID = {}
        self.num_reviewerIDs = 0

    def addUser(self, reviewerID):
        for id_ in reviewerID:
            self.reviewerID2index[id_] = self.num_reviewerIDs
            self.index2reviewerID[self.num_reviewerIDs] = id_
            self.num_reviewerIDs += 1   

class Preprocess:
    def __init__(self, hidden_size=300, setence_max_len=80):

        self.setence_max_len = setence_max_len
        self.hidden_size = hidden_size
        self.unknown_ctr = 0

    def indexesFromSentence(self, voc, sentence , MAX_LENGTH = 200):
        sentence_segment = sentence.split(' ')[:MAX_LENGTH]
        return [voc.word2index[word] for word in sentence_segment]

    def indexesFromSentence_Evaluate(self, voc, sentence , MAX_LENGTH = 200):
        sentence_segment = sentence.split(' ')[:MAX_LENGTH]

        indexes = list()
        for word in sentence_segment:
            try:
                indexes.append(voc.word2index[word])
            except KeyError as ke:
                indexes.append(PAD_token)
                self.unknown_ctr +=1
            except Exception as msg:
                print('Exception :\n', msg)

        return indexes

    def _inputVar(self, l, voc, testing=False):
        """Returns padded input sequence tensor and lengths"""
        if(testing):
            indexes_batch = [self.indexesFromSentence_Evaluate(voc, sentence, MAX_LENGTH=self.setence_max_len) for sentence in l]
        else:   # for training
            indexes_batch = [self.indexesFromSentence(voc, sentence, MAX_LENGTH=self.setence_max_len) for sentence in l]

        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        padList = self._zero_padding(indexes_batch)
        padVar = torch.LongTensor(padList)
        return padVar, lengths

    def _batch2TrainData(self, myVoc, sentences, rating, isSort = False, normalizeRating = False, testing=False, useNLTK=True):
        sentences_rating_pair = list()
        for index in range(len(sentences)):
            sentences_rating_pair.append(
                [
                    sentences[index], rating[index], 
                ]
            )
        
        sentences_batch, rating_batch = [], []
        for pair in sentences_rating_pair:
            sentences_batch.append(pair[0])
            rating_batch.append(pair[1])

        inp, lengths = self._inputVar(
            sentences_batch,
            myVoc,
            testing
            )
        
        label = torch.tensor([val for val in rating_batch])
        
        # Is normalize rating
        if(normalizeRating):
            label = (label-1)/(5-1)

        return inp, lengths, label

    def _batch2LabelData(self, rating, this_asin, this_reviewerID, itemObj, userObj, normalizeRating = False):
        sentences_rating_pair = list()
        for index in range(len(rating)):
            sentences_rating_pair.append(
                [
                    rating[index],
                    this_asin[index], this_reviewerID[index]
                ]
            )

        rating_batch, this_asin_batch, this_reviewerID_batch  = [], [], []
        for pair in sentences_rating_pair:
            rating_batch.append(pair[0])
            this_asin_batch.append(itemObj.asin2index[pair[1]])
            this_reviewerID_batch.append(userObj.reviewerID2index[pair[2]])

        
        label = torch.tensor([val for val in rating_batch])
        
        # Is normalize rating
        if(normalizeRating):
            label = (label-1)/(5-1)

        # asin and reviewerID batch association
        asin_batch = torch.tensor([val for val in this_asin_batch])
        reviewerID_batch = torch.tensor([val for val in this_reviewerID_batch])

        return label, asin_batch, reviewerID_batch

    def GenerateTrainingBatches(self, USERorITEM, candidateObj, voc, 
        net_type = 'item_base', start_of_reviews=0, num_of_reviews = 5 , 
        batch_size = 5, testing=False, get_rating_batch=False):
        """
            Create sentences & length encoding

            if net_type is user-base, then candidateObj should be itemObj
            elif net_type is item-base, then candidateObj should be userObj
        """

        new_training_batches_sentences = list()
        new_training_batches_ratings = list()
        new_training_batches = list()
        new_training_batches_asins = list()
        num_of_batch_group = 0
        

        # for each user numberth reivews
        
        for review_ctr in range(start_of_reviews, num_of_reviews, 1):
            new_training_batch_sen = dict() # 40 (0~39)
            new_training_batch_rating = dict()
            new_training_batch_asin = dict()
            training_batches = dict()
            
            for user_ctr in tqdm.tqdm(range(len(USERorITEM))):
                
                # Over sampling 0611
                if(not True and USERorITEM[user_ctr].rating[review_ctr]==1 ):
                    _os_count = 3
                else:
                    _os_count = 1

                for _os_c in range(_os_count):   

                    # Insert group encodeing
                    if((user_ctr % batch_size == 0) and user_ctr>0):
                        num_of_batch_group+=1
                        
                        # encode pre group
                        training_batch = self._batch2TrainData(
                                            voc, 
                                            new_training_batch_sen[num_of_batch_group-1], 
                                            new_training_batch_rating[num_of_batch_group-1],
                                            isSort = False,
                                            normalizeRating = False,
                                            testing=testing
                                            )
                        # training_batches[num_of_batch_group-1].append(training_batch)
                        training_batches[num_of_batch_group-1] = training_batch

                    this_user_sentence = USERorITEM[user_ctr].sentences[review_ctr]
                    this_user_rating = USERorITEM[user_ctr].rating[review_ctr]
                    
                    # Using itemObj/userObj to find index
                    if(net_type == 'user_base'):
                        # origin user_base : candidate would be product
                        this_user_asin = USERorITEM[user_ctr].this_asin[review_ctr]
                        this_user_asin_index = candidateObj.asin2index[this_user_asin]
                    elif(net_type == 'item_base'):
                        # item_base : candidate would be user
                        this_asin_user = USERorITEM[user_ctr].this_reviewerID[review_ctr]
                        this_asin_user_index = candidateObj.reviewerID2index[this_asin_user]


                    if(num_of_batch_group not in new_training_batch_sen):
                        new_training_batch_sen[num_of_batch_group] = []
                        new_training_batch_rating[num_of_batch_group] = []
                        new_training_batch_asin[num_of_batch_group] = []
                        # training_batches[num_of_batch_group] = []

                    new_training_batch_sen[num_of_batch_group].append(this_user_sentence)
                    new_training_batch_rating[num_of_batch_group].append(this_user_rating)   
                    
                    if(net_type == 'user_base'):
                        new_training_batch_asin[num_of_batch_group].append(this_user_asin_index) 
                    elif(net_type == 'item_base'):                
                        new_training_batch_asin[num_of_batch_group].append(this_asin_user_index) 
                    

                    # Insert group encodeing (For Last group)
                    if(user_ctr == (len(USERorITEM)-1)):
                        num_of_batch_group+=1
                        # encode pre group
                        training_batch = self._batch2TrainData(
                                            voc, 
                                            new_training_batch_sen[num_of_batch_group-1], 
                                            new_training_batch_rating[num_of_batch_group-1],
                                            isSort = False,
                                            normalizeRating = False,
                                            testing=testing                                    
                                            )
                        # training_batches[num_of_batch_group-1].append(training_batch)
                        training_batches[num_of_batch_group-1] = training_batch            


            new_training_batches_sentences.append(new_training_batch_sen)
            new_training_batches_ratings.append(new_training_batch_rating)
            new_training_batches.append(training_batches)
            new_training_batches_asins.append(new_training_batch_asin)

            num_of_batch_group = 0

            # print('Unknown word:{}'.format(self.unknown_ctr))
        if get_rating_batch:
            return new_training_batches, new_training_batches_asins, new_training_batches_ratings
        else:
            return new_training_batches, new_training_batches_asins       

    def GenerateBatchLabelCandidate(self, labels_, asins_, reviewerIDs_, batch_size, CANDIDATE, candidateObj, voc, start_of_reviews=5, num_of_reviews = 1, testing=False, mode='', net_type='item_base'):
        num_of_batch_group = 0
        batch_labels = dict()
        candidate_asins = dict()
        candidate_reviewerIDs = dict()

        batch_labels[num_of_batch_group] = list()
        candidate_asins[num_of_batch_group] = list()
        candidate_reviewerIDs[num_of_batch_group] = list()

        for idx in range(len(labels_)):
            if((idx % batch_size == 0) and idx > 0):
                num_of_batch_group+=1
                batch_labels[num_of_batch_group] = list()
                candidate_asins[num_of_batch_group] = list()
                candidate_reviewerIDs[num_of_batch_group] = list()
            
            batch_labels[num_of_batch_group].append(labels_[idx])
            candidate_asins[num_of_batch_group].append(asins_[idx])
            candidate_reviewerIDs[num_of_batch_group].append(reviewerIDs_[idx])

        if(mode =='generate'):
            testing_batches, testing_asin_batches = self.GenerateTrainingBatches(CANDIDATE, candidateObj, voc, start_of_reviews=start_of_reviews,
                net_type=net_type,
                num_of_reviews = start_of_reviews+num_of_reviews , batch_size = batch_size, testing=testing)
            
            return batch_labels, candidate_asins, candidate_reviewerIDs, testing_batches
        
        else:
            return batch_labels, candidate_asins, candidate_reviewerIDs


    def _generate_label_encoding(self, USER, num_of_reviews, num_of_rating, itemObj, userObj):
        """Generate & encode candidate batch"""

        training_labels = dict()
        training_asins = dict()
        training_reviewerIDs = dict()

        for user_ctr in range(len(USER)):
            new_training_label = list()
            new_training_asin = list()
            new_training_reviewerID = list()

            # Create label (rating, asin, reviewers) structure
            for rating_ctr in range(num_of_reviews, num_of_reviews+num_of_rating, 1):
                
                # append candidate label
                tmp = USER[user_ctr]
                this_traning_label = USER[user_ctr].rating[rating_ctr]
                new_training_label.append(this_traning_label)

                # append candidate asin
                this_asin = USER[user_ctr].this_asin[rating_ctr]
                new_training_asin.append(this_asin)

                # append candidate user
                this_reviewerID = USER[user_ctr].this_reviewerID[rating_ctr]
                new_training_reviewerID.append(this_reviewerID)
            
            # Get batch of data
            new_training_label, asin_batch, reviewerID_batch = self._batch2LabelData(new_training_label, new_training_asin, new_training_reviewerID, itemObj, userObj)
            
            # Record data by user counter
            training_labels[user_ctr] = new_training_label   
            training_asins[user_ctr] = asin_batch   
            training_reviewerIDs[user_ctr] = reviewerID_batch  

        return training_labels, training_asins, training_reviewerIDs

    #########  Beload is after rescontruct

    def _unicode_to_ascii(self, s):
        """convert all letters to lowercase """
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def _normalize_string(self, s):
        """Lowercase, trim, and remove non-letter characters"""
        s = self._unicode_to_ascii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()

        # <num>
        # s = re.sub(r"([.!?])", r" \1", s)
        # s = re.sub(r"[^a-zA-Z0-9.!?]+", r" ", s)
        # s = re.sub(r"(\d+)", r" <number>", s)
        # s = re.sub(r"\s+", r" ", s).strip()
    
        return s

    def _zero_padding(self, l, fillvalue=PAD_token):
        return list(itertools.zip_longest(*l, fillvalue=fillvalue))

    def _read_asin_reviewer(self, dpath='HANN-Plus/data/asin_reviewer_list/', table='clothing_'):
        with open('{}{}asin.csv'.format(dpath, table),'r') as file:
            content = file.read()
            asin = content.split(',')
            print('asin count : {}'.format(len(asin)))

        with open('{}{}reviewerID.csv'.format(dpath, table),'r') as file:
            content = file.read()
            reviewerID = content.split(',')
            print('reviewerID count : {}'.format(len(reviewerID)))
        
        return asin, reviewerID

    def get_train_set(self, CANDIDATE, itemObj, userObj, voc, batchsize=32, num_of_reviews=5, num_of_rating=1, net_type='item_base', testing=False, mode=''):
        batch_labels = list()
        candidate_asins = list()
        candidate_reviewerIDs = list()
        label_sen_batch = None

        for idx in range(0, num_of_rating, 1):
            # Generate train set
            training_labels, training_asins, training_reviewerIDs = self._generate_label_encoding(
                CANDIDATE, 
                num_of_reviews+idx, 1, 
                itemObj, 
                userObj
                )
            
            if(net_type =='item_base'):
                candidateObj = userObj
            elif(net_type =='user_base'):
                candidateObj = itemObj

            # Train set to batch data
            if(mode==''):
                _labels, _asins, _reviewerIDs = self.GenerateBatchLabelCandidate(
                    training_labels, 
                    training_asins, 
                    training_reviewerIDs, 
                    batchsize,
                    CANDIDATE, 
                    candidateObj, 
                    voc,
                    testing=testing,
                    mode=mode,
                    net_type=net_type
                    )
            elif (mode=='generate'):
                _labels, _asins, _reviewerIDs, _label_sen_batches = self.GenerateBatchLabelCandidate(training_labels, 
                    training_asins, 
                    training_reviewerIDs, 
                    batchsize,
                    CANDIDATE, 
                    candidateObj, 
                    voc,
                    testing=testing,
                    mode=mode,
                    net_type=net_type
                    )
                label_sen_batch = _label_sen_batches   
            
            batch_labels.append(_labels)
            candidate_asins.append(_asins)
            candidate_reviewerIDs.append(_reviewerIDs)
            
        
        return batch_labels, candidate_asins, candidate_reviewerIDs, label_sen_batch

    def generate_candidate_voc(self, res, having_interaction=6, generate_voc=True, user_based=True, net_type = 'item_base'):
        """This method is used to generate vocab. from DB sentences according user-base or item-base"""

        print('\nCreating Voc ...')
        st = time.time()
          
        CANDIDATE = list()
        candiate2index = dict()
        last_candidate = ''
        ctr = -1
        
        if(net_type == 'user_base'):
            ROW_NAME = 'reviewerID'     # SQL sort by reviewerID
        elif(net_type == 'item_base'):
            ROW_NAME = 'asin'           # SQL sort by asin
                    
        # Creating voc.
        if(generate_voc):
            myVoc = Voc('Review')

        for index in tqdm.tqdm(range(len(res))):
            # encounter a new candidate
            if(last_candidate != res[index][ROW_NAME]):
                last_candidate = res[index][ROW_NAME]               # record new candidate id

                if(net_type == 'user_base'):
                    CANDIDATE.append(ReviewerHistory(res[index][ROW_NAME])) # append a object
                elif(net_type == 'item_base'):
                    CANDIDATE.append(AsinHistory(res[index][ROW_NAME])) # append a object

                # CANDIDATE.append(AsinHistory(res[index][ROW_NAME])) # append a object
                ctr += 1                                            # adding counter
            
            # Dealing with sentences
            if(res[index]['rank'] < having_interaction + 1):
                if(res[index]['reviewText'] == None):
                    current_sentence = self._normalize_string('')
                else:
                    current_sentence = self._normalize_string(res[index]['reviewText'])
                
                if(generate_voc):
                    myVoc.addSentence(current_sentence) # myVoc add word 
                
                CANDIDATE[ctr].add_data(
                            current_sentence,
                            res[index]['overall'], 
                            res[index]['asin'],
                            res[index]['reviewerID']
                        )
                candiate2index[res[index][ROW_NAME]] = ctr  # store index

        print('CANDIDATE length:[{}]'.format(len(CANDIDATE)))
        print('Voc creation complete. [{}]'.format(time.time()-st))
        
        if(generate_voc):
            return myVoc, CANDIDATE, candiate2index
        else:
            return CANDIDATE, candiate2index


    def generate_voc(self, res):
        """
        The method is for generating voc 
        """
        st = time.time()
        print('Creating Voc ...') 
        myVoc = Voc('Review')
            
        for index in tqdm.tqdm(range(len(res))):            
            current_sentence = self._normalize_string(res[index]['reviewText'])
            myVoc.addSentence(current_sentence) # myVoc add word 

        print('Voc creation complete. [{}]'.format(time.time()-st))
        return myVoc


    def load_data(self, sqlfile='', mode='train', table='clothing_', rand_seed=42, test_on_train_data=False, num_of_generative=None):
        """Load dataset from database"""

        print('\nLoading asin/reviewerID from cav file...')
        asin, reviewerID = self._read_asin_reviewer(table=table)
        print('Loading asin/reviewerID complete.')

        # asin/reviewerID to index
        itemObj = item()
        itemObj.addItem(asin)
        userObj = user()
        userObj.addUser(reviewerID)

        print('\nLoading dataset from database...') 
        st = time.time()
        
        # Loading SQL cmd from .sql file
        with open(sqlfile) as file_:
            if(sqlfile!='' and mode=='train'):
                sql_cmd = file_.read().split(';')[0]
            elif(mode=='validation'):
                if(test_on_train_data):
                    sql_cmd = file_.read().split(';')[1].replace('NOT IN','IN')
                else:
                    sql_cmd = file_.read().split(';')[1]
            elif(mode=='test'):
                if(test_on_train_data):
                    sql_cmd = file_.read().split(';')[2].replace('NOT IN','IN')
                else:
                    sql_cmd = file_.read().split(';')[2]                    


        # Setup random seed
        sql_cmd = sql_cmd.replace(
            "RAND()", 
            "RAND({})".format(rand_seed)
            )

        # Setup generative table
        if(num_of_generative!=None):
            sql_cmd = sql_cmd.replace(
                "&swap&", 
                "{}".format(num_of_generative)
                )
        
        print("""####################################\nSQL Command:\n{}\n####################################\n""".format(sql_cmd))

        # Conn. to db & select data
        conn = DBConnection()
        res = conn.selection(sql_cmd)
        conn.close()

        print('Loading complete. [{}]'.format(time.time()-st))
        
        return res, itemObj, userObj

    def load_pretain_word(self, voc, pretrain_words):
        weights_matrix = np.zeros((voc.num_words, self.hidden_size))
        words_found = 0

        for index, word in voc.index2word.items():
            if(word == 'PAD'):
                weights_matrix[index] = np.zeros(self.hidden_size)   
            else:
                try: 
                    weights_matrix[index] = pretrain_words[word]
                    words_found += 1
                except KeyError as msg:
                    weights_matrix[index] = np.random.uniform(low=-1, high=1, size=(self.hidden_size))
                    print(msg)
        
        return weights_matrix

    def load_asin2title(self, sqlfile='HANN-Plus/SQL/cloth_asin2title.sql')->dict:
        """Load dataset from database"""

        # Loading SQL cmd from .sql file
        with open(sqlfile) as file_:
            sql_cmd = file_.read()
        
        print("""####################################\nSQL Command:\n{}\n####################################\n""".format(sql_cmd))

        # Conn. to db & select data
        conn = DBConnection()
        res = conn.selection(sql_cmd)
        conn.close()

        asin2title = dict()
        for _val in res:
            asin2title[_val['asin']] = _val['title']

        return asin2title
    
    def load_sparsity_reviews(self, fpath):
        """Method for loading sparsity reviews id."""
        # reload a file to a variable
        # with open('{}/review_sparsity_{}.pickle'.format(fpath, fname),  'rb') as file:
        #     can2sparsity = pickle.load(file)
        with open(fpath,  'rb') as file:
            can2sparsity = pickle.load(file)        

        return can2sparsity
import torch
import torch.nn as nn
import torch.nn.functional as F

class IntraReviewGRU(nn.Module):
    def __init__(self, hidden_size, embedding, itemEmbedding, userEmbedding, n_layers=1, dropout=0, latentK = 64, method = 'dualFC'):
        super(IntraReviewGRU, self).__init__()
        
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = embedding
        self.itemEmbedding = itemEmbedding
        self.userEmbedding = userEmbedding
        self.method = method
    

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
            self.linear_alpha = torch.nn.Linear(hidden_size, 1) 

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

        elif self.method == 'dualFC':
            self.linear1 = torch.nn.Linear(hidden_size, hidden_size)
            self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
            self.linear_alpha = torch.nn.Linear(hidden_size, 1)       

        self.intra_review = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), 
                          bidirectional=True)
                         
    def CalculateAttn(self, key_vector, query_vector):
        
        # Calculate weighting score
        if(self.method == 'dualFC'):
            x = F.relu(self.linear1(key_vector) +
                    self.linear2(query_vector) 
                )
            weighting_score = self.linear_alpha(x)
            # Calculate attention score
            intra_attn_score = torch.softmax(weighting_score, dim = 0)

        elif (self.method=='dot'):
            intra_attn_score = key_vector * query_vector
            
        elif (self.method=='general'):
            energy = self.attn(query_vector)
            x = F.relu(key_vector * energy)
            weighting_score = self.linear_alpha(x)
            # Calculate attention score            
            intra_attn_score = torch.softmax(weighting_score, dim = 0)

        return intra_attn_score
        
    def forward(self, input_seq, input_lengths, item_index, user_index, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)           
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, enforce_sorted=False)
        # Forward pass through GRU
        outputs, hidden = self.intra_review(packed, hidden)
 
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]

        # Calculate element-wise product
        elm_w_product = self.itemEmbedding(item_index) * self.userEmbedding(user_index)

        # Calculate attention score

        if self.method != 'without':
            intra_attn_score = self.CalculateAttn(outputs, elm_w_product)
            new_outputs = intra_attn_score * outputs
            intra_outputs = torch.sum(new_outputs , dim = 0)    # output sum
        else:
            intra_outputs = torch.sum(outputs , dim = 0)    # output sum
            intra_attn_score = None

        # Return output and final hidden state
        return intra_outputs, hidden, intra_attn_score

class HANNiNet(nn.Module):
    def __init__(self, hidden_size, embedding, iEmbedding, uEmbedding, 
        n_layers=1, dropout=0, latentK = 64, concat_rating=False):
        super(HANNiNet, self).__init__()
        
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.latentK = latentK

        self.embedding = embedding
        self.iEmbedding = iEmbedding
        self.uEmbedding = uEmbedding

        self.concat_rating = concat_rating

        self.attn = nn.Linear(self.hidden_size, hidden_size)
        self.linear_beta = torch.nn.Linear(hidden_size, 1)
 
        GRU_InputSize = hidden_size

        if(self.concat_rating):
            GRU_InputSize = GRU_InputSize+5                

        self.inter_review = nn.GRU(GRU_InputSize, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout))
                          
        self.dropout = nn.Dropout(dropout)
        
        self.fc_doubleK = nn.Linear(hidden_size*2 , self.latentK*2)
        self.fc_singleK = nn.Linear(self.latentK*2, self.latentK)
        self.fc_out = nn.Linear(self.latentK, 1)
    
    def CalculateAttn(self, key_vector, query_vector):
        # Calculate weighting score
        energy = self.attn(query_vector)
        x = F.relu(key_vector * energy)
        weighting_score = self.linear_beta(x)
        # Calculate attention score            
        inter_attn_score = torch.softmax(weighting_score, dim = 0)

        return inter_attn_score

    def forward(self, intra_outputs, this_candidate_index, item_index, user_index, hidden=None, review_rating=None):
        
        inter_input = intra_outputs

        if(self.concat_rating):
            inter_input = torch.cat((inter_input, review_rating), 2)

        # Forward pass through GRU
        outputs, hidden = self.inter_review(inter_input, hidden)

        # Calculate element-wise product
        elm_w_product_inter = self.iEmbedding(item_index) * self.uEmbedding(user_index)

        # Calculate attention score
        inter_attn_score = self.CalculateAttn(outputs, elm_w_product_inter)

        # Consider attention score
        weighting_outputs = inter_attn_score * outputs
        context_vector = weighting_outputs
        outputs = torch.sum(weighting_outputs , dim = 0)  

        # Concat. interaction vector & GRU output
        outputs_cat = torch.cat((outputs, elm_w_product_inter), dim=1)
        
        # dropout
        outputs_drop = self.dropout(outputs_cat)

        # Multiple layer to decrease dimension
        outputs_ = self.fc_doubleK(outputs_drop)
        outputs_ = self.fc_singleK(outputs_)
        outputs_ = self.fc_out(outputs_)

        sigmoid_outputs = torch.sigmoid(outputs_)
        sigmoid_outputs = sigmoid_outputs.squeeze(0)

        # Return output and final hidden state
        return sigmoid_outputs, hidden, inter_attn_score, context_vector

class DecoderGRU(nn.Module):
    def __init__(self, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(DecoderGRU, self).__init__()

        # self.dec_merge_rating = dec_merge_rating
        self._use_coverage = False

        # Keep for reference
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)

        # Drop out
        self._dropout = nn.Dropout(dropout)

        self.weight_coverage = nn.Linear(output_size, hidden_size)
        self._fc_300 = nn.Linear(905, hidden_size)

        # GRU model
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.out = nn.Linear(hidden_size, output_size)

        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)

        self.concat_coverage = nn.Linear(hidden_size * 3, hidden_size)
    
    def set_user_embedding(self, _user_embedding):
        self._user_embedding = _user_embedding

    def set_item_embedding(self, _item_embedding):
        self._item_embedding = _item_embedding

    def CalculateAttn(self, hidden, encoder_output):
        # Linear layer to calculate weighting score
        energy = self.attn(encoder_output)
        weighting_score = torch.sum(hidden * energy, dim=2)
        weighting_score = weighting_score.t()
        
        # Activation function
        attn_weights = torch.softmax(weighting_score, dim=1).unsqueeze(1)

        return attn_weights
    
    def get_softmax_output(self):
        return torch.softmax(self.output, dim=1)
    
    def set_coverage_prob(self, previous_prob, _use_coverage):
        self._use_coverage = _use_coverage
        self.coverage_prob = previous_prob
        pass

    def forward(self, input_step, last_hidden, context_vector, 
        _encode_rating=None , _user_emb=None, _item_emb=None, _enable_attention=True):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        
        # Get user/item embedding of current input user/item
        _user_emb = self._user_embedding(_user_emb)
        _user_emb = _user_emb.unsqueeze(0)

        _item_emb = self._item_embedding(_item_emb)
        _item_emb = _item_emb.unsqueeze(0)


        # Construct initial decoder hidden state
        input_hidden = torch.tanh(
            self._fc_300(torch.cat((
                last_hidden , 
                _user_emb, 
                _item_emb, 
                _encode_rating
                ), dim=2))
        )
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, input_hidden)
        rnn_output = rnn_output.squeeze(0)

        if(_enable_attention):
            attn_weights = self.CalculateAttn(rnn_output, context_vector)
            
            context = attn_weights.bmm(context_vector.transpose(0, 1))
            context = context.squeeze(1)

            # Concat. rnn output & context inf.
            concat_input = torch.cat((rnn_output, context), 1)   
            # Drop out layer
            concat_input = self._dropout(concat_input) 

            if (self._use_coverage):
                # Considerate coverage mechanism
                _coverage_rep = self.weight_coverage(self.coverage_prob)    # previous word probability sum.
                # Concat. attention output & coverage representation
                concat_input = torch.cat((concat_input, _coverage_rep.squeeze_(0)), 1 )
                # FC & Activation
                concat_output = torch.tanh(self.concat_coverage(concat_input))          
                output = self.out(concat_output)
                pass
            else:
                # Origin : without coverage
                concat_output = torch.tanh(self.concat(concat_input))
                output = self.out(concat_output)
                pass
        else:
            # Without attention from Encoder
            rnn_output = torch.tanh(rnn_output)
            output = self.out(rnn_output)
        
        self.output = output

        # log softmax
        output = self.logsoftmax(output)

        # Return output and final hidden state
        return output, hidden, attn_weights

class SubNetwork(nn.Module):
    def __init__(self, hidden_size, embedding, itemEmbedding, userEmbedding, 
        n_layers=1, dropout=0, latentK = 64, 
        concat_rating = False, 
        netType='item_base', method='dualFC'):
        super(SubNetwork, self).__init__()
        
        self.method = method
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.latentK = latentK

        self.embedding = embedding
        self.itemEmbedding = itemEmbedding
        self.userEmbedding = userEmbedding

        self.concat_rating = concat_rating
        self.netType = netType

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
            self.linear_beta = torch.nn.Linear(hidden_size, 1)   
            
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

        elif self.method == 'dualFC':            
            self.linear3 = torch.nn.Linear(hidden_size, hidden_size)
            self.linear4 = torch.nn.Linear(hidden_size, hidden_size)
            self.linear_beta = torch.nn.Linear(hidden_size, 1)      

        GRU_InputSize = hidden_size

        if(self.concat_rating):
            GRU_InputSize = GRU_InputSize+5

        self.inter_review = nn.GRU(GRU_InputSize, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout))
                          
        self.dropout = nn.Dropout(dropout)
        
        self.fc_doubleK = nn.Linear(hidden_size*2 , self.latentK*2)
        self.fc_singleK = nn.Linear(self.latentK*2, self.latentK)
        self.fc_out = nn.Linear(self.latentK, 1)
    
    def CalculateAttn(self, key_vector, query_vector):
        
        # Calculate weighting score
        if(self.method == 'dualFC'):
            x = F.relu(self.linear3(key_vector) +
                    self.linear4(query_vector) 
                )
            weighting_score = self.linear_beta(x)
            # Calculate attention score
            inter_attn_score = torch.softmax(weighting_score, dim = 0)

        elif (self.method=='dot'):
            inter_attn_score = key_vector * query_vector
            
        elif (self.method=='general'):
            energy = self.attn(query_vector)
            x = F.relu(key_vector * energy)
            weighting_score = self.linear_beta(x)
            # Calculate attention score            
            inter_attn_score = torch.softmax(weighting_score, dim = 0)

        return inter_attn_score

    def forward(self, inter_input, item_index, user_index, hidden=None, review_rating=None):
        
        if(self.concat_rating):
            inter_input = torch.cat((inter_input, review_rating), 2)

        # Forward pass through GRU
        outputs, hidden = self.inter_review(inter_input, hidden)

        # Calculate element-wise product
        elm_w_product_inter = self.itemEmbedding(item_index) * self.userEmbedding(user_index)

        # Calculate attention score
        inter_attn_score = self.CalculateAttn(outputs, elm_w_product_inter)

        # Consider attention score
        weighting_outputs = inter_attn_score * outputs
        outputs_sum = torch.sum(weighting_outputs , dim = 0)  

        return outputs_sum, hidden, inter_attn_score

class PredictionLayer(nn.Module):
    def __init__(self, hidden_size, itemEmbedding, userEmbedding, dropout=0, latentK = 64):
        super(PredictionLayer, self).__init__()
        
        self.hidden_size = hidden_size
        self.latentK = latentK

        self.itemEmbedding = itemEmbedding
        self.userEmbedding = userEmbedding
        self.dropout = nn.Dropout(dropout)
        
        self.fc_doubleK = nn.Linear(hidden_size*3 , self.latentK*2)
        self.fc_singleK = nn.Linear(self.latentK*2, self.latentK)
        self.fc_out = nn.Linear(self.latentK, 1)
    
    def forward(self, q_i, q_u, item_index, user_index):
        
        # Calculate element-wise product
        elm_w_product_inter = self.itemEmbedding(item_index) * self.userEmbedding(user_index)

        # Concat. interaction vector & GRU output
        rep_cat = torch.cat((q_i, q_u), dim=1)
        rep_cat = torch.cat((rep_cat, elm_w_product_inter), dim=1)
        
        # dropout
        rep_cat = self.dropout(rep_cat)

        # hidden_size to 2*K dimension
        outputs_ = self.fc_doubleK(rep_cat)
        # 2*K to K dimension
        outputs_ = self.fc_singleK(outputs_)

        # K to 1 dimension
        outputs_ = self.fc_out(outputs_)
        sigmoid_out = torch.sigmoid(outputs_)
        outputs = sigmoid_out.squeeze(0)

        # Return output and final hidden state
        return outputs

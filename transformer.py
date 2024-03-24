# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 08:23:15 2024

@author: user
"""

import torch
import torch.nn as nn
import math 

#word embedding 
class InputEmbeddings(nn.Module):
    #d_model: dimension of the model d_model include 128, 256, 512, or even larger values
    #number_of_psi_unique:how many there is psi unique in level 1 ( if w use many features it become how many there is psi unique+how many there is kappa unique  )
    def __init__(self, d_model: int, number_of_psi_unique: int):
        super().__init__()
        self.d_model=d_model
        self.number_of_psi_unique=number_of_psi_unique
        self.embedding=nn.Embedding(number_of_psi_unique,d_model)
       
    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)
 
    
#positionnal encoding
#d_model the size of the vector of the vector of positionnal encoding 
#sec_length the maximun number of psi_1 in all the patient  ( of the sentence )
#dropout to make the model less overfit 
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_psi: int, dropout: float) -> None:
        super().__init__()
        #max_psi:Maximum number of psi values in all patient 
        self.d_model = d_model
        self.max_psi = max_psi
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(max_psi, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, max_psi, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)
        #what is the buffer : to save the tensor as long the file aloong with the state of the model 
        
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)
    
    
#the normalisation layer of the encdoer  of the decoder (ADD& Norm ) 
# if we have a batch of 3 items ( 3 samples of PPG) we calculate in each sample mean and std
# alpha and biais  used in training alpha  is for multiplication and bias  to add
class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter multiply
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter adding 

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    

#feed forword layer of the encoder 
# 2 matrices W1 W2  it's done by nn.liner()
#d_ff : the size of the feedforward neural network layer used in the transformer's encoder
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x)))) 
        
        
#build of the Multihead attention : 
#Input --> Q K V we multiplae by  Wq Wk Wv we apply the attention fonction for each head 
# for now we are going to do one head 
# we multiply the result of the attention function by Wo

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        #h=number of heads and d_model should bee divisible by h  
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq d_model X d_model
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk d_model X d_model
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv d_model X d_model
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo d_model X d_model
        self.dropout = nn.Dropout(dropout)
    
    #we define a static method wich means we can call this function:static methods do not operate on specific instances of the class;
    # to call this method we Nameclass.namefunction(name_input)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]# it's the last size of the tensor querry 
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)# @ matrix application in python 
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)#replace all the number witch mask==0 by -1e9 witch mean -infini so when we apply softmax result = 0 
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores #the first one is givin to the next layer and the seconde is udes for visualisation (representing data or information graphically : what is the score for each interraction.)

    def forward(self, q, k, v, mask):
        #mask : if we wonts some psi not to interract with others 
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # In this step we devide each query and value to give it to each head 
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> by transpose (batch, h, seq_len, d_k)
        #tensor.view() changing tensor  shape while preserving the total number of elements.
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)#we multiply W_o*x
    
    
    
#define the connections : there are 3 connection :
#1/between the normalisation layer_1(before feedforward )  and the feedforward layer 
#2/beftween the normalizzation layyer_1  and the other normalisation layer( after feedforward )
#3/between the feedforwrd and the normalisation layer_2 
# w etake the input(a) we skip one layer and give it to the one after than we took the output of the prvios layer 
# and give it to the layer after but we combine it with the input (a) 

class ResidualConnection(nn.Module):
    
        def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)
    
        def forward(self, x, sublayer):
            #sublayer is the previous layer 
            #self.norm(x) apply a normalisation layer to the tensor x
            # sublayer():we apply the previous layer to the outut of the normalisation layer 
            #x+ .. :the original input tensor x is added to the output of the dropout operation
            return x + self.dropout(sublayer(self.norm(x)))
        
        
#now we are going to develop the connection between two blowks (Add & Norm /feed forward ) and (Add & Norm /Multihead attention )
#the big block that is repeated N times where the output of j is sent to j+1 (j=2..N-1) and the ouput fo the last block is our output tha we need 
#this block will contain one Multihead attention 2 Add&Norm and one Feed forward 
class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        #self attention : Q K and V = x 
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        #residual_connections list of module 
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        
        # x--> multihead attention and X skipp Multihead attention and goes to Add&Norm and than we do the connection betweeen Multihead attention and ADD&norm
        #lambda x: self.self_attention_block(x, x, x, src_mask): aply the self attenttion (Q=V=K=x each psi is interacting with other psi in the same sample ) 
        # self.self_attention_block : it's the forword method of the MultiHeadAttentionBlock class
        #x and the self attention is connected by ResidualConnection  
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) 
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
# the finale layer that convert the output of the encoder of the transformer into the
# the position of psi (projectiong the embeding into the list of psi)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, number_of_psi_unique) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, number_of_psi_unique)

    def forward(self, x) -> None:
        # (batch, max_psi, d_model) --> (batch, max_psi, number_of_psi_unique)
        return self.proj(x)

#now we define the archtiture of a transformer 

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, src_embed: InputEmbeddings, src_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    
    def build_transformer(src_number_of_psi_unique: int , src_max_psi: int , d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048):
        # Create the embedding layers
        src_embed = InputEmbeddings(d_model, src_number_of_psi_unique)
    
        # Create the positional encoding layers
        src_pos = PositionalEncoding(d_model, src_max_psi, dropout)
        
        # Create the encoder blocks
        encoder_blocks = []
        for _ in range(N):
            encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
            feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
            encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
            encoder_blocks.append(encoder_block)
    
        
        # Create the encoder
        encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
        
        # Create the projection layer
        projection_layer = ProjectionLayer(d_model,src_number_of_psi_unique )
        
        # Create the transformer
        transformer = Transformer(encoder, src_embed, src_pos, projection_layer)
        
        # Initialize the parameters
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        return transformer
    

    
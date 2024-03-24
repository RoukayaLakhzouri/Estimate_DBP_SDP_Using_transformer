# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 21:04:45 2024

@author: user
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas



#we create the data set that contains the features
class FeaturesDataset(Dataset):

    def __init__(self, ds_features, tokenizer, max_psi):
        #ds_features: contains all psi_N1  /max of  number of psi in a sample (number of fetures )
        super().__init__()
        self.max_psi = max_psi

        self.ds_features = ds_features
        self.tokenizer = tokenizer


        self.sos_token_psi = torch.tensor(tokenizer.token_to_id([-1 for i in range(1201) ]))
        self.eos_token_psi = torch.tensor(tokenizer.token_to_id([0 for i in range(1201) ]))
        self.pad_token_psi = torch.tensor(tokenizer.token_to_id([-10 for i in range(1201) ]))

    def __len__(self):
        return len(self.ds_features)

    def __getitem__(self, idx):
        # indice        
        #psi_matrix1 = self.ds_features.dataset.iloc[idx,0] # used if the input data has one colones 
        psi_matrix1 = self.ds_features.dataset.iloc[idx,0]
        Liste_psi1 = [ psi_matrix1[:,i] for i in range(psi_matrix1.shape[1])]        

        # Transform the text into tokens

        enc_input_tokens = self.tokenizer.encode_tokenizer_psi(Liste_psi1)

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.max_psi - len(enc_input_tokens)   # We will add <s> and </s>

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 :
            raise ValueError("Sentence is too long")

        # Add <sos> and <eoss> token
        #concatinate 3 tensors 
        
        encoder_input = torch.cat(
            [
                #self.sos_token_psi,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                #self.eos_token_psi,
                torch.tensor([self.pad_token_psi] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )
        label =encoder_input.clone() #the model's goal is to predict the label sequence.

        print(" encoder_input length :  \n:",encoder_input.shape)

        

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.max_psi
        assert label.size(0) == self.max_psi

        return {
            "encoder_input": encoder_input,  # (max_psi)
            #we built a masck that are Padding are not used in attention mecanisme
            "encoder_mask": (encoder_input != self.pad_token_psi).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "label": label,  # (seq_len)
            "Liste_psi1": Liste_psi1
        }
    
def causal_mask(size):
    #this method will make only the features before are seen  
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


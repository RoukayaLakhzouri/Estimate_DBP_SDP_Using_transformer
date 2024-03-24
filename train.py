# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 06:32:16 2024

@author: user
"""
import json

import torch
import torch.nn as nn
import pandas as pd
import numpy as np 
from transformer import Transformer

from torch.utils.data import Dataset, DataLoader,random_split

#from torch.optim.lr_scheduler import LambdaLR

from torch.utils.tensorboard import SummaryWriter
import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer

from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer# create the vocabulary liste 
from tokenizers.pre_tokenizers import Whitespace #split the sentence into word 

from dataset import FeaturesDataset

from config import get_weights_file_path,latest_weights_file_path,get_config

num_chif=3#nombre de chiffre prii apres virgue 

colonne_indices_pics_N1= pd.read_csv('Features\colonne_indices_pics_N1.csv')
colonne_indices_pics_N2=pd.read_csv('Features\colonne_indices_pics_N2.csv')
colonne_indices_pics_N3=pd.read_csv('Features\colonne_indices_pics_N3.csv')

colonne_kappa_N1=pd.read_csv('Features\colonne_kappa_N1.csv')
colonne_kappa_N2=pd.read_csv('Features\colonne_kappa_N2.csv')
colonne_kappa_N3=pd.read_csv('Features\colonne_kappa_N3.csv')

colonne_valeurs_pics_N1=pd.read_csv('Features\colonne_valeurs_pics_N1.csv')
colonne_valeurs_pics_N2=pd.read_csv('Features\colonne_valeurs_pics_N2.csv')
colonne_valeurs_pics_N3=pd.read_csv('Features\colonne_valeurs_pics_N3.csv')

index_pics=pd.read_csv('Features\index_pics.csv')

#faire un appel de la colonne 1 sans NAN de type array :
#colonne_indices_pics_N1.iloc[0,0:index_pics.iloc[0,0]].values

tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))


    
def create_psinnor_kappa(sample):
    N1=index_pics.iloc[sample,0]#nombre de psi Niveau 1 
    N2=index_pics.iloc[sample,1]#nombre de psi Niveau 2
    N3=index_pics.iloc[sample,2]#nombre de psi Niveau 3
    
    psinnor_N1=np.zeros((1201,N1+2))
    psinnor_N2=np.zeros((1201,N2+2))
    psinnor_N3=np.zeros((1201,N3+2))
    
    Kappa_N1=np.zeros((N1+2,1))
    Kappa_N2=np.zeros((N2+2,1))
    Kappa_N3=np.zeros((N3+2,1))    
    
    psinnor_N1[:,0]=np.ones(1201)*-1 #psi relative a 'SOS'
    psinnor_N2[:,0]=np.ones(1201)*-1 #psi relative a 'SOS'
    psinnor_N3[:,0]=np.ones(1201)*-1 #psi relative a 'SOS'
    
    Kappa_N1[0]=-5 # kappa relative a 'SOS'
    Kappa_N2[0]=-5 # kappa relative a 'SOS'
    Kappa_N3[0]=-5  # kappa relative a 'SOS'     
    
    for j in range(1,max(N1,N2,N3)+2):
        if(j<N1+1):
            indice_pic_1=int(colonne_indices_pics_N1.iloc[sample,j-1]-1)
            #indice_pic_1=int(colonne_indices_pics_N1.iloc[sample,0:index_pics.iloc[sample,0]][j])
            psinnor_N1[indice_pic_1,j]=round(colonne_valeurs_pics_N1.iloc[sample,j-1],num_chif)
            Kappa_N1[j]=round(colonne_kappa_N1.iloc[sample,j-1],num_chif)
            
        if(j<N2+1):
            indice_pic_N21=int(colonne_indices_pics_N2.iloc[2*sample,j-1]-1)
            #indice_pic_N21=int(colonne_indices_pics_N2.iloc[2*sample-1,0:0:index_pics.iloc[sample,1]-1][j])
            psinnor_N2[indice_pic_N21,j]=round(colonne_valeurs_pics_N2.iloc[2*sample,j-1],num_chif)
            Kappa_N2[j]=round(colonne_kappa_N2.iloc[sample,j-1],num_chif)
            
            indice_pic_N22=int(colonne_indices_pics_N2.iloc[2*sample+1,j-1]-1)        
            psinnor_N2[indice_pic_N22,j]=round(colonne_valeurs_pics_N2.iloc[2*sample+1,j-1],num_chif)
            
        if(j<N3+1):
            indice_pic_N31=int(colonne_indices_pics_N3.iloc[3*sample,j-1]-1)
            psinnor_N3[indice_pic_N31,j]=round(colonne_valeurs_pics_N3.iloc[3*sample,j-1],num_chif)
            
            indice_pic_N32=int(colonne_indices_pics_N3.iloc[3*sample+1,j-1]-1)
            psinnor_N3[indice_pic_N32,j]=round(colonne_valeurs_pics_N3.iloc[3*sample+1,j-1],num_chif)
            Kappa_N3[j]=round(colonne_kappa_N3.iloc[sample,j-1],num_chif)

            
            indice_pic_N33=int(colonne_indices_pics_N3.iloc[3*sample+2,j-1]-1)
            psinnor_N3[indice_pic_N33,j]=round(colonne_valeurs_pics_N3.iloc[3*sample+2,j-1],num_chif)
    return psinnor_N1,psinnor_N2,psinnor_N3,Kappa_N1,Kappa_N2,Kappa_N3



psinnor_N1,psinnor_N2,psinnor_N3,Kappa_N1,Kappa_N2,Kappa_N3=create_psinnor_kappa(2)


def create_feature_data(Num_sample): #0--> Num_sample-1
    Liste_psi_N1=[] #Créer une liste vide pour contenir tous  les psi de niveau 1 
    Liste_psi_N2=[] #Créer une liste vide pour contenir tous  les psi de niveau 2 
    Liste_psi_N3=[] #Créer une liste vide pour contenir tous  les psi de niveau 3 

    Liste_K_N1=[] #Créer une liste vide pour contenir tous  les Kappa de niveau 1
    Liste_K_N2=[] #Créer une liste vide pour contenir tous  les kappa de niveau 2   
    Liste_K_N3=[] #Créer une liste vide pour contenir tous  les kappa de niveau 3
    
    
    for sample in range(Num_sample):
        psinnor_N1,psinnor_N2,psinnor_N3,Kappa_N1,Kappa_N2,Kappa_N3=create_psinnor_kappa(sample)
        
        Liste_psi_N1.append(psinnor_N1)
        Liste_psi_N2.append(psinnor_N2)
        Liste_psi_N3.append(psinnor_N3)
        
        Liste_K_N1.append(Kappa_N1)
        Liste_K_N2.append(Kappa_N2)
        Liste_K_N3.append(Kappa_N3)


    df=pd.DataFrame({'psi niveau 1':Liste_psi_N1,'psi niveau 2':Liste_psi_N2,'psi niveau 3':Liste_psi_N3,'Kappa niveau 1':Liste_K_N1,'Kappa niveau 2':Liste_K_N2,'Kappa niveau 3':Liste_K_N3})
    return df
ds_features=create_feature_data(5)

psinnor = ds_features['psi niveau 1'][2]
Liste_psi1 = [ psinnor_N1[:,i] for i in range(psinnor_N1.shape[1])]


#function builds and saves a tokenizer based on the provided feature data this function is like Build_tokenizer
def create_Dictionary(Num_sample): 
    Liste_psi=[] #Créer une liste vide pour contenir tous  les psi 
    Liste_K=[] 
    for sample in range(Num_sample):
        psi_matrix1,psi_matrix2,psi_matrix3,Kappa_N1,Kappa_N2,Kappa_N3=create_psinnor_kappa(sample)
        
        Liste_psi1 = [ psi_matrix1[:,i] for i in range(1,psi_matrix1.shape[1]-1)]
        #Liste_psi2 = [ psi_matrix2[:,i] for i in range(1,psi_matrix2.shape[1]-1)]
        #Liste_psi3 = [ psi_matrix3[:,i] for i in range(1,psi_matrix3.shape[1]-1)]
        
        # Ajouter les vecteurs de chaque liste à la liste concaténée
        Liste_psi.extend(Liste_psi1)
        #Liste_psi.extend(Liste_psi2)
        #Liste_psi.extend(Liste_psi3)
        
        Liste_K.extend(Kappa_N1[1:psi_matrix1.shape[1]-1])
        #Liste_K.extend(Kappa_N2[1:psi_matrix2.shape[1]-1])
        #Liste_K.extend(Kappa_N3[1:psi_matrix3.shape[1]-1])
        
        
        
    # Convertir la liste de vecteurs en un ensemble de tuples
    ensemble_unique = set(tuple(vecteur) for vecteur in Liste_psi)
    ensemble_unique1=set(tuple(k) for k in Liste_K)
    
    # Convertir l'ensemble de tuples en une liste de vecteurs uniques
    liste_psi_uniques = [list(tup) for tup in ensemble_unique]
    liste_psi_uniques.insert(0,[0*i for i in range(1201) ])#ajouter tokens associer a 'EOS'
    liste_psi_uniques.insert(0,[-1 for i in range(1201) ])#ajouter tokens associer a 'SOS'
    liste_psi_uniques.insert(0,[-10 for i in range(1201) ])#ajouter tokens associer a 'PAD'
    
    
    liste_k_uniques = [list(tup) for tup in ensemble_unique1]
    liste_k_uniques.insert(0,[0]) #ajouter tokens associer a 'EOS'
    liste_k_uniques.insert(0,[-5]) #ajouter tokens associer a 'SOS'
    liste_k_uniques.insert(0,[-10]) #ajouter tokens associer a 'PAD'
    
    
    
    df=pd.DataFrame({'tokens_psi':liste_psi_uniques})
    df1=pd.DataFrame({'tokens_kappa':liste_k_uniques})
    
    df['valeur associer psi']=[i for i in range(len(liste_psi_uniques))]# associe a chaque psi un entier 
    df['valeur associer psi'][0]=1 #associer un ID en 'PAD'   
    df['valeur associer psi'][1]=2 #associer un ID en 'SOS'
    df['valeur associer psi'][2]=3 #associer un ID en 'EOS'
        
    df1['valeur associer kappa']=[i for i in range(len(liste_psi_uniques)+1,len(liste_psi_uniques)+len(liste_k_uniques)+1)] # associe a chaque kappa  un entier 
    
    df1['valeur associer kappa'][0]=len(liste_psi_uniques)+1    
    df1['valeur associer kappa'][1]=len(liste_psi_uniques)+2
    df1['valeur associer kappa'][2]=len(liste_psi_uniques)+3
    
    return df,df1
df,df1=create_Dictionary(4)


class Tokenizer():
    
    def __init__(self,data_psi,data_kappa):
        self.tokens_psi = data_psi
        self.tokens_kappa = data_kappa
    
    def token_to_id(self,token):
        if (len(token)==1):
            series_lignes =self.tokens_kappa['tokens_kappa'] 
            indice_trouve = None
            for indice, ligne in enumerate(series_lignes):
                print(ligne)
                if ligne == list(token):
                    indice_trouve = indice
                    break
            id=self.tokens_kappa['valeur associer kappa'][indice_trouve]
        else :
            series_lignes =self.tokens_psi['tokens_psi'] 
            indice_trouve = None
            for indice, ligne in enumerate(series_lignes):
                if ligne == list(token):
                    indice_trouve = indice
                    break
            id=self.tokens_psi['valeur associer psi'][indice_trouve]
        return id
    
    def encode_tokenizer_psi(self, list_Features):
        # Encode input list_of features of psi  into token IDs
        encoding = [self.token_to_id(feature) for feature in list_Features]
        return encoding
    
    def encode_tokenizer_kappa(self,list_Features):
        # Encode input list_of features  into token IDs
        encoding = [self.token_to_id(feature) for feature in list_Features]
        return encoding
    
    def get_Num_psi_unique(self):
        return self.tokens_psi.shape[0]

    def get_Num_K_unique(self):
        return self.tokens_kappa.shape[0]
    def save_psi(self, file_path):
        # Convert DataFrame to dictionary
        vocab_dict = self.tokens_psi.to_dict()
    
        # Save dictionary to JSON file
        with open(file_path, 'w') as file:
            json.dump(vocab_dict, file)
    def save_kappa(self, file_path):
        # Convert DataFrame to dictionary
        vocab_dict = self.tokens_kappa.to_dict()
    
        # Save dictionary to JSON file
        with open(file_path, 'w') as file:
            json.dump(vocab_dict, file)
 
        
 
#this function take the number of samples     
def get_ds(Num_sample):
    ds_raw=create_feature_data(Num_sample)
     # It only has the train split, so we divide it overselves
    ds_psi,ds_kappa=create_Dictionary(Num_sample)
     # Build tokenizers
    tokenizer = Tokenizer(ds_psi, ds_kappa)
    
    max_psi=max([np.shape(ds_raw['psi niveau 1'][sample])[1] for sample in range(Num_sample) ])# nombre de psi max dans niveau 1 
    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * Num_sample)
    val_ds_size = Num_sample - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = FeaturesDataset(train_ds_raw, tokenizer,max_psi)
    val_ds = FeaturesDataset(val_ds_raw, tokenizer,max_psi)

    # Find the maximum length of each sentence in the source and target sentence
    max_len = 0

    for item in ds_raw['psi niveau 1']:
        src_ids = tokenizer.encode_tokenizer_psi([item[:,j] for j in range(np.shape(item)[1])])
        max_len = max(max_len, len(src_ids))

    print(f'Max length of source sentence: {max_len}')
    

    train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer,max_psi





Num_sample=5
ds_raw=create_feature_data(Num_sample)
ds_raw1=pd.DataFrame(ds_raw.iloc[:,0])
ds_psi,ds_kappa=create_Dictionary(Num_sample)
  # Build tokenizers
tokenizer = Tokenizer(ds_psi, ds_kappa)

max_psi=max([np.shape(ds_raw['psi niveau 1'][sample])[1] for sample in range(Num_sample) ])# nombre de psi max dans niveau 1 
# Keep 90% for training, 10% for validation
train_ds_size = int(0.9 * Num_sample)
val_ds_size = Num_sample - train_ds_size
train_ds_raw, val_ds_raw = random_split(ds_raw1, [train_ds_size, val_ds_size])

train_ds = FeaturesDataset(train_ds_raw, tokenizer,max_psi)
val_ds = FeaturesDataset(val_ds_raw, tokenizer,max_psi)
dict=train_ds.__getitem__(4)

# Find the maximum length of each sentence in the source and target sentence
max_len = 0

for item in ds_raw['psi niveau 1']:
    src_ids = tokenizer.encode_tokenizer_psi([item[:,j] for j in range(np.shape(item)[1])])
    max_len = max(max_len, len(src_ids))

# print(f'Max length of source sentence: {max_len}')


train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=True)
val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)





def get_model(Num_psi_unique,max_psi):
    model = Transformer.build_transformer(Num_psi_unique,max_psi)
    return model



# train_dataloader, val_dataloader, tokenizer,max_psi = get_ds(5)
# device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
# device=torch.device(device)
# for i in range(train_dataloader.batch_size) :
#     break
# print({k:v.shape for k,v in train_dataloader[i].items()})


    model_filename = get_weights_file_path(config, f"{epoch:02d}")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step
    }, model_filename)

# dataloader_iterator = iter(train_dataloader)
# #batch=next()
# i=0
# while True:
#     try:
#         print("\n\n\n dans boucle \n:",i)
#         batch = next(dataloader_iterator)
#         # Process the batch here
#         i=i+1
#         print("New i",i)
#     except StopIteration:
#         break


def train_model(config,Num_sample):
    #Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer,max_psi = get_ds(Num_sample)
    tokenizer_path = Path(config['tokenizer_file'])
    tokenizer.save_psi(str(tokenizer_path)) 
    
    model = get_model(tokenizer.get_Num_psi_unique(),max_psi).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name']) #create log files in the runs/tmodel directory to write various kinds of data to the log files,

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9) #set up an optimizer that will update the parameters of the model during training using the Adam optimization algorithm
    
    # # In this part we will impliment the code that can help us to restore the state of the model and the state of the optimizer
    # # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        #model_filename=get_weights_file_path(config,preload)
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id([-10 for i in range(1201) ]), label_smoothing=0.1).to(device)#label_smoothing: to less overfit and to increase accrucy
    
    #training loop
    
    for epoch in range(initial_epoch, config['num_epochs']):
        #release all unoccupied cached memory currently held by the CUDA memory
        torch.cuda.empty_cache() 
        
    #     #before training the model to activate specific training-related behaviors
        model.train() 
        
        #tqdm library: for tracking the progress of loops or iterations 
        #train_dataloader: used to load batches of data during training (iterate over datasets, providing features like batching, shuffling, and multiprocessing.)
        #batch_iterator : This variable stores the tqdm-wrapped iterator returned
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        print ('batch_iterator=',type(batch_iterator))
        for batch in batch_iterator:
            print('batch',type(batch))
            encoder_input = batch['encoder_input'].to(device) # (b, max_psi)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, max_psi)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, max_psi, d_model)
            proj_output = model.project(encoder_output) # (B, seq_len, max_psi)
            
            
            # Compare the output with the label
            label = batch['label'].to(device) # (B, max_psi)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer.get_Num_psi_unique(), label.view(-1)))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        #run_validation(model, val_dataloader, tokenizer, max_psi, device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config,10)
    
    
    
    # device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    # device=torch.device(device)
    # model = get_model(tokenizer.get_Num_psi_unique(),max_psi).to(device)
    # # Tensorboard
    # writer = SummaryWriter(config['experiment_name']) #create log files in the runs/tmodel directory to write various kinds of data to the log files,
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9) #set up an optimizer that will update the parameters of the model during training using the Adam optimization algorithm
    
    # # # In this part we will impliment the code that can help us to restore the state of the model and the state of the optimizer
    # # # If the user specified a model to preload before training, load it
    # initial_epoch = 0
    # global_step = 0    
    # preload = config['preload']
    # model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    # if model_filename:
    #     #model_filename=get_weights_file_path(config,preload)
    #     print(f'Preloading model {model_filename}')
    #     state = torch.load(model_filename)
    #     model.load_state_dict(state['model_state_dict'])
    #     initial_epoch = state['epoch'] + 1
    #     optimizer.load_state_dict(state['optimizer_state_dict'])
    #     global_step = state['global_step']
    # else:
    #     print('No model to preload, starting from scratch')    
    
    # loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id([-10 for i in range(1201) ]), label_smoothing=0.1).to(device)#label_smoothing: to less overfit and to increase accrucy
    # for epoch in range(initial_epoch, config['num_epochs']):
    #     batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
    #     for batch in train_dataloader :
    #         #dict1={k:v. for k,v in batch.items()}
    #         #print("dict1\n",dict1)
    #         print("the error is in encoder_input \n ")
    #         encoder_input = batch['encoder_input'].to(device) # (b, max_psi)
            
    #         print("the error is in decoder_input \n ")
    #         encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, max_psi)
        
    #         # Run the tensors through the encoder, decoder and the projection layer
    #         encoder_output = model.encode(encoder_input, encoder_mask) # (B, max_psi, d_model)
    #         proj_output = model.project(encoder_output) # (B, seq_len, max_psi)
            
            
    #         # Compare the output with the label
    #         label = batch['label'].to(device) # (B, max_psi)
        
    #         # Compute the loss using a simple cross entropy
    #         loss = loss_fn(proj_output.view(-1, tokenizer.get_Num_psi_unique(), label.view(-1)))
    #         batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
        
    #         # Log the loss
    #         writer.add_scalar('train loss', loss.item(), global_step)
    #         writer.flush()
        
    #         # Backpropagate the loss
    #         loss.backward()
        
    #         # Update the weights
    #         optimizer.step()
    #         optimizer.zero_grad(set_to_none=True)
        
    #         global_step += 1
        
    #         # Run validation at the end of every epoch
    #         #run_validation(model, val_dataloader, tokenizer, max_psi, device, lambda msg: batch_iterator.write(msg), global_step, writer)
        
    #         # Save the model at the end of every epoch

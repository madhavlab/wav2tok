import torch
import torch.nn.functional as F
import torch.utils.data as data
import pickle

import random
import numpy as np

import librosa


import torch.nn as nn

from torch.optim.lr_scheduler import LambdaLR






def save_weights(model,epoch, name):

    #####    IN : model, epoch_number , name of the save file
    #####    FUNCTION: Saves model weights at {epoch} EPOCH as {name}.bin




     return torch.save(model.state_dict(), f'../weights/{name}_{epoch}.bin')




def load_weights(trainer, epoch,  name):

    #####    IN : model, epoch_number , name of the save file
    #####    FUNCTION: loads model weights at {epoch} EPOCH from {name}.bin

    print("||| Weights Loaded |||")

    map = {'cuda:1': 'cuda:0'}
    return trainer.load_state_dict(torch.load(f'../weights/{name}_{epoch}.bin', map_location = map))


def save(data, filename):

    #####    IN : data , name of the save file
    #####    FUNCTION: saves file as {filename}.bin



     with open(f'../bin/{filename}.bin', 'wb') as fp:
              pickle.dump(data, fp)
              print('done')
     print("|||||||||||||||||SAVED||||||||||||||||")
     return


def load(filename):

    #####    IN :  name of the  file
    #####    FUNCTION: loads {filename}.bin

     with open(f'../bin/{filename}.bin', 'rb') as fp:
             data= pickle.load( fp)
     return data




def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return   max(
           0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
       )

    return LambdaLR(optimizer, lr_lambda, last_epoch)




def data_balancer(wav_dict_dir = 'WAV_DICTS'):
    train , valid, test = load(wav_dict_dir)
    train_ , valid_  = {}, {}

    train_len , valid_len  = [], []

    sign = np.array([-1,1] )
 
    step = np.arange(1,4)
    #pitch_scale = pitch_scale[np.where( pitch_scale != 0)[0]]
    
    for keys in train.keys():
          train_len.append(len(train[keys]))
          valid_len.append(len(valid[keys]))
         

    max_len_train = max(train_len)
    max_len_valid = max(valid_len)
    
    for keys in train.keys():
         el_train = train[keys]
         el_valid = valid[keys]
         el_train_ , el_valid_ = el_train , el_valid
         if len(el_train) < max_len_train:
            for i in range(max_len_train - len(el_train)):
                   random.seed(42)
                   
                   el_train_.append(librosa.effects.pitch_shift(random.sample(el_train,1)[0],\
                                   sr = 16000, n_steps = np.random.choice(sign)* np.random.random()))
                   
         if len(el_valid) < max_len_valid:
            for i in range(max_len_valid - len(el_valid)):
                   random.seed(42)
                   el_valid_.append(librosa.effects.pitch_shift(random.sample(el_valid,1)[0],\
                                   sr = 16000, n_steps = np.random.choice(sign)* np.random.random())) 
      
         train_[keys] = el_train_

         valid_[keys] = el_valid_

    list  = [train_, valid_ , test]
     
    save(list, 'WAV_DICTS_BALANCED')


    return list
    

def CQT_gen(wav_dict_dir = 'WAV_DICTS_BALANCED'):

     train , valid, test = load(wav_dict_dir)
     train_ , valid_ , test_ = {}, {}, {}





     for keys in train.keys():

         print(keys)
         train_[keys] = [np.abs(librosa.cqt(i, sr= 16000) ) for i in train[keys]]

         valid_[keys] = [np.abs(librosa.cqt(i, sr= 16000)) for i in valid[keys]]

         test_[keys] = [np.abs(librosa.cqt(i, sr= 16000)) for i in test[keys]]
     
     list = [train_ , valid_ , test_ ]

     save(list, 'CQT_DICTS_BALANCED')
      
     return list


def FEAT_gen(wav_dict_dir = 'WAV_DICTS'):

     train , valid, test = load(wav_dict_dir)
     train_ , valid_ , test_ = {}, {}, {}

     train1_ , valid1_ , test1_ = {}, {}, {}




     for keys in train.keys():

         print(keys)
         train_[keys] = [np.abs(librosa.cqt(i,n_bins = 83,hop_length = 256, sr= 8000) ) for i in train[keys]] 

         valid_[keys] = [np.abs(librosa.cqt(i,n_bins = 83,hop_length = 256, sr= 8000)) for i in valid[keys]]  

         test_[keys] = [np.abs(librosa.cqt(i,n_bins = 83,hop_length = 256, sr= 8000)) for i in test[keys]]

         train1_[keys] = [np.abs(librosa.stft(i, n_fft= 256) ) for i in train[keys]]

         valid1_[keys] = [np.abs(librosa.stft(i, n_fft= 256)) for i in valid[keys]]

         test1_[keys] = [np.abs(librosa.stft(i, n_fft= 256)) for i in test[keys]]

     list = [train_ , valid_ , test_ ]
     list2 = [train1_ , valid1_ , test1_]
     save(list, 'CQT_DICTS')
     save(list2, 'STFT_DICTS')
     return list, list2



def dataset_gen(song_dict,):
  X = []
  num_el = 0
  for keys in song_dict.keys():
      list_ = list(song_dict[keys])

      for i in list_ :
                      
              X.append((i,keys))


  print(f'\nLength of DATASET  = {len(X)}\n')
  #save(X, dataset_name )
  return X





class DATA:





       def __init__(self,  X, dic = 0):
            self.X = X
            assert (
            dic in [0,1]
             ), f"no dictionaries available => 0-> train , 1 -> test"
            self.wav_dict = load("CQT_DICTS")[dic] #CQT_DICTS_TIME_STRETCHED2 CQT_DICTS_BALANCED
            
             
                              



       def __len__(self):
              return len(self.X)

       def __getitem__(self,item):

              set = self.X[item]
              x1 = set[0]
              key = set[1]
              #print(key)
              random.seed(42)
              x2 = random.sample([i for i in self.wav_dict[key] \
                        ],1)[0]
         
              
             


              return {

                     0: torch.Tensor(np.abs(x1)).permute(1,0),
                     1: torch.Tensor(np.abs(x2)).permute(1,0),

                       }


class DATA_:





       def __init__(self,  X, dic = 0):
            self.X = X
            assert (
            dic in [0,1]
             ), f"no dictionaries available => 0-> train , 1 -> test"
            self.wav_dict = load('SP_DICT')[dic]   # load("STFT_DICTS")[dic] #CQT_DICTS_TIME_STRETCHED2 CQT_DICTS_BALANCED






       def __len__(self):
              return len(self.X)

       def __getitem__(self,item):

              set = self.X[item]
              x1 = set[0]
              key = set[1]
              #print(key)
              random.seed(42)
              x2 = random.sample([i for i in self.wav_dict[key] \
                        ],1)[0]





              return {

                     0: torch.Tensor(np.abs(x1)).permute(1,0),
                     1: torch.Tensor(np.abs(x2)).permute(1,0),

                       }

class DATA_NEW:





       def __init__(self,  X, dic = 0):
            self.X = X
            assert (
            dic in [0,1]
             ), f"no dictionaries available => 0-> train , 1 -> test"
            self.wav_dict = load("STFT_DICTS")[dic] #CQT_DICTS_TIME_STRETCHED2 CQT_DICTS_BALANCED






       def __len__(self):
              return len(self.X)

       def __getitem__(self,item):

              set = self.X[item]
              x1 = set[0]
              key = set[1]
              #print(key)

              new_keys = [i for i in self.wav_dict.keys() if i != key]

              random.seed(42)

              dis_key = random.sample(new_keys, 1)[0]

             # random.seed(42)

              #x2 = random.sample([i for i in self.wav_dict[key] \
               #         ],1)[0]

              random.seed(42)

              x3 = random.sample([i for i in self.wav_dict[dis_key]],1)[0]





              return {

                     0: torch.Tensor(np.abs(x1)).permute(1,0),
                #     1: torch.Tensor(np.abs(x2)).permute(1,0),
                     1: torch.Tensor(np.abs(x3)).permute(1,0)
                       }



class DATA3_:





       def __init__(self,  X, dic = 0):
            self.X = X
            assert (
            dic in [0,1]
             ), f"no dictionaries available => 0-> train , 1 -> test"
            self.wav_dict = load("DATASET_STFT")[dic] #CQT_DICTS_TIME_STRETCHED2 CQT_DICTS_BALANCED






       def __len__(self):
              return len(self.X)

       def __getitem__(self,item):

              set = self.X[item]
              x1 = set[0]
              key = set[1]
              #print(key)
              random.seed(42)
              x2 = random.sample([i for i in self.wav_dict \
                        ],1)[0][0]
#              print(len(x2), x2[0].shape)




              return {

                     0: torch.Tensor(np.abs(x1)).permute(1,0),
                     1: torch.Tensor(np.abs(x2)).permute(1,0),

                       }

class DATA2_:





       def __init__(self,  X, dic = 0):
            self.X = X
            assert (
            dic in [0,1]
             ), f"no dictionaries available => 0-> train , 1 -> test"
            self.wav_dict = load("NEW_DICT")[dic] #CQT_DICTS_TIME_STRETCHED2 CQT_DICTS_BALANCED






       def __len__(self):
              return len(self.X)

       def __getitem__(self,item):

              set = self.X[item]
              x1 = set[0]
              key = set[1]
              #print(key)
              random.seed(42)
              x2 = self.wav_dict[key] #random.sample([i for i in self.wav_dict[key] \
                   #     ],1)[0]
#              print(len(x2), x2[0].shape)




              return {

                     0: torch.Tensor(np.abs(x1)).permute(1,0),
                     1: torch.Tensor(np.abs(x2)).permute(1,0),

                       }



class DATA2:





       def __init__(self,  X , dic = None):
            self.X = X
            
             
                              



       def __len__(self):
              return len(self.X)

       def __getitem__(self,item):

              set = self.X[item]
              x1 = set[0]
              
              x2 = set[1]
              
             # print(x1.shape)
              #print(x2.shape)             


              return {

                     0: torch.Tensor(x1).permute(1,0),
                     1: torch.Tensor(x2),

                       }







class DATA3:





       def __init__(self,  X, dic = 0):
            self.X = X






       def __len__(self):
              return len(self.X)

       def __getitem__(self,item):

              set = self.X[item]
              x1 = set

           

              return { 

                     0: torch.Tensor(x1),
                     #1: torch.Tensor(x2),

                       }





def similar_dataset_gen(song_dict,num_limit =15 ):
  X = []
  num_el = 0
  for keys in song_dict.keys():
      list_ = list(song_dict[keys])
      
      for i in list_ :
              random.seed(42)
              lis = [l for l in list_ if np.sum(np.subtract(l,i)) != 0]
              others = lis if len(lis) < num_limit else random.sample(lis, num_limit) 

              for el in others:

                   X.append((i,el))
    

  print(f'\nLength of DATASET  = {len(X)}\n')
  #save(X, dataset_name )
  return X


def data_gen(song_dict,num_limit =15 ):
  X = []
  num_el = 0
  for keys in song_dict.keys():
      list_ = list(song_dict[keys])

      for i in list_ :
             X.append(i)


  print(f'\nLength of DATASET  = {len(X)}\n')
  #save(X, dataset_name )
  return X

#if __name__ == "__main__":




def stretch_and_squeeze(wav = 'WAV_DICTS'):
     train , valid, test = load(wav) 
     train_s , valid_s , test_stft = {}, {}, {}
     train_c , valid_c, test_cqt  = {}, {}, {}


     for keys in train.keys():

         print(keys)
         #print( max(0.5,np.random.random()*1.5 ))
         #train_s[keys] = [np.abs(librosa.cqt( librosa.effects.time_stretch(i, rate = max(0.5, 0.5 + np.random.random()*1.5)) , sr= 8000) ) for i in train[keys]]

         #valid_s[keys] = [np.abs(librosa.cqt(librosa.effects.time_stretch(i, rate = max(0.5,0.5+  np.random.random()*1.5) ) , sr= 8000) ) for i in valid[keys]]

         te = [librosa.effects.time_stretch(i, rate = max(0.5,0.5 + np.random.random()*1.5 ))  for  i in test[keys]]


         test_cqt[keys] = [np.abs(librosa.cqt(i,n_bins = 83,hop_length = 256, sr = 8000) )  for i in te]
         test_stft[keys] = [np.abs(librosa.stft(i,n_fft = 256 )) for i in te]


        # train_c[keys] = [np.abs(librosa.cqt(librosa.effects.time_stretch(i, rate = 2 ), sr= 16000) ) for i in train[keys]]

         #valid_c[keys] = [np.abs(librosa.cqt(librosa.effects.time_stretch(i, rate = 2 ), sr= 16000) )  for i in valid[keys]]

         #test_c[keys] = [np.abs(librosa.cqt(librosa.effects.time_stretch(i, rate = 2 ), sr= 16000) )  for i in test[keys]]




    # list_s = [train_s , valid_s ]

     #save(list_s, 'CQT_DICTS_TIME_STRETCHED')

     #list_c = [train_c , valid_c  ]

     #save(list_c, 'CQT_DICTS_COMPRESSED')
     save(test_stft, 'TEST_STFT_DICTS_TIME_STRETCHED')
     save(test_cqt, 'TEST_CQT_DICTS_TIME_STRETCHED')


     return






def stretch_and_squeezeOLD(wav = 'WAV_DICTS'):
     train , valid, test = load(wav)
     train_s , valid_s , test_s = {}, {}, {}
     train_c , valid_c, test_c  = {}, {}, {}
            

     for keys in train.keys():

         print(keys)
         #print( max(0.5,np.random.random()*1.5 ))
         train_s[keys] = [np.abs(librosa.cqt( librosa.effects.time_stretch(i, rate = max(0.5, 0.5 + np.random.random()*1.5)) , sr= 16000) ) for i in train[keys]]

         valid_s[keys] = [np.abs(librosa.cqt(librosa.effects.time_stretch(i, rate = max(0.5,0.5+  np.random.random()*1.5) ) , sr= 16000) ) for i in valid[keys]]

         test_s[keys] = [np.abs(librosa.cqt(librosa.effects.time_stretch(i, rate = max(0.5,0.5 + np.random.random()*1.5 )), sr= 16000) )  for i in test[keys]]

        # train_c[keys] = [np.abs(librosa.cqt(librosa.effects.time_stretch(i, rate = 2 ), sr= 16000) ) for i in train[keys]]

         #valid_c[keys] = [np.abs(librosa.cqt(librosa.effects.time_stretch(i, rate = 2 ), sr= 16000) )  for i in valid[keys]]

         #test_c[keys] = [np.abs(librosa.cqt(librosa.effects.time_stretch(i, rate = 2 ), sr= 16000) )  for i in test[keys]]



           
     list_s = [train_s , valid_s ]

     save(list_s, 'CQT_DICTS_TIME_STRETCHED')

     #list_c = [train_c , valid_c  ]

     #save(list_c, 'CQT_DICTS_COMPRESSED')
     save(test_s, 'TEST_CQT_DICTS_TIME_STRETCHED')
    # save(test_c, 'TEST_CQT_DICTS_COMPRESSED')
 

     return list_s


def stretch_and_squeeze_stft(wav = 'WAV_DICTS_BALANCED'):
     train , valid, test = load(wav)
     train_s , valid_s , test_s = {}, {}, {}
     train_c , valid_c, test_c  = {}, {}, {}


     for keys in train.keys():

         print(keys)
         #print( max(0.5,np.random.random()*1.5 ))
         train_s[keys] = [np.abs(librosa.stft( librosa.effects.time_stretch(i, rate = max(0.5, 0.5 + np.random.random()*1.5)) , sr= 16000) ) for i in train[keys]]

         valid_s[keys] = [np.abs(librosa.cqt(librosa.effects.time_stretch(i, rate = max(0.5,0.5+  np.random.random()*1.5) ) , sr= 16000) ) for i in valid[keys]]

         test_s[keys] = [np.abs(librosa.cqt(librosa.effects.time_stretch(i, rate = max(0.5,0.5 + np.random.random()*1.5 )), sr= 16000) )  for i in test[keys]]

        # train_c[keys] = [np.abs(librosa.cqt(librosa.effects.time_stretch(i, rate = 2 ), sr= 16000) ) for i in train[keys]]

         #valid_c[keys] = [np.abs(librosa.cqt(librosa.effects.time_stretch(i, rate = 2 ), sr= 16000) )  for i in valid[keys]]

         #test_c[keys] = [np.abs(librosa.cqt(librosa.effects.time_stretch(i, rate = 2 ), sr= 16000) )  for i in test[keys]]




     list_s = [train_s , valid_s ]

     save(list_s, 'CQT_DICTS_TIME_STRETCHED2')

     #list_c = [train_c , valid_c  ]

     #save(list_c, 'CQT_DICTS_COMPRESSED')
     save(test_s, 'TEST_CQT_DICTS_TIME_STRETCHED2')
    # save(test_c, 'TEST_CQT_DICTS_COMPRESSED')


     return list_s




def compress(wav = 'WAV_DICTS'):
     train , valid, test = load(wav)
     train_s , valid_s , test_s = {}, {}, {}
     train_c , valid_c, test_c  = {}, {}, {}


     for keys in train.keys():

         print(keys)
         #print( max(0.5,np.random.random()*1.5))
         train_s[keys] = [np.abs(librosa.cqt( librosa.effects.time_stretch(i, rate =2) , sr= 16000) ) for i in train[keys]]

         valid_s[keys] = [np.abs(librosa.cqt(librosa.effects.time_stretch(i, rate = 2 ), sr= 16000 )) for i in valid[keys]]

        # test_s[keys] = [np.abs(librosa.cqt(librosa.effects.time_stretch(i, rate = 2, sr= 16000) )  for i in test[keys]]

    



     list_s = [train_s , valid_s ]

     save(list_s, 'CQT_DICTS_COMPRESSED')

     #list_c = [train_c , valid_c  ]

     #save(list_c, 'CQT_DICTS_COMPRESSED')
    # save(test_s, 'TEST_CQT_DICTS_TIME_STRETCHED')
    # save(test_c, 'TEST_CQT_DICTS_COMPRESSED')


     return list_s


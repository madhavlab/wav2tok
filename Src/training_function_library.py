import numpy as np

import torch 

import torch.nn as nn 

import torch.nn.functional as F


from new_function_library import load , save 

import torch.utils.data as data

import random

import librosa

def gen_data_from_dict(dict_):

   X = []
   for i in dict_.keys():
   
       el = dict_[i]
       
       for j in el:
          
          X.append((j, i))
          
   return X
          

def collate_function(batch):

    data1 = [item[0] for item in batch]
    data2 = [item[1] for item in batch]
    return [data1, data2]          
          
          
def collate_function2(batch):

    data1 = [item[0] for item in batch]
    data2 = [item[1] for item in batch]
    data3 = [item[2] for item in batch]    
    
    return [data1, data2, data3]            
    
    
def collate_function3(batch):

    data1 = [item[0] for item in batch]
   
    
    return [data1]             

class DATA:

    def __init__(self, dataset ,same_length = False,is_triplet = False,\
                                                  is_dict =True, single = False, train = 0 ):
    
         
          if is_dict:
                    
                      
             self.dict = load(dataset)[train]
             
             self.X = gen_data_from_dict(self.dict)
             
          else:
          
             self.X = load(dataset)[train]
          
               
          
          
    def __len__(self):
 
        
          return len(self.X)

    def __getitem__(self,item):
    
          if is_dict:
          
             set = self.X[item]
             
             
             utt = set[0]
             
             key = set[1]
             
             
             if single:
              
                return {'0': torch.tensor(utt)}
             
             
             utt2 = random.sample(self.dict[key],1)[0]
             
             

             
             if is_triplet:
             
                 nkey = random.sample([i for i in self.dict.keys() if i != key], 1)[0]
                 
                 utt3 = random.sample(self.dict[nkey],1)[0]
                 
                 
             
             if same_length :
             
               
                 
                 len1 = len(utt)
                 
                 len2 = len(utt2)
                 
                 if is_triplet:
                 
                    len3 = len(utt3)
                 
                 if len1 != len2 :
                 
                   len_ = (len1 + len2)/2
                   
                   if is_triplet:
                   
                      len_ = (len1+ len2 + len3)/2  
                 
                   utt = librosa.effects.time_stretch(utt, rate = len_/len1)
                 
                   utt2 = librosa.effects.time_stretch(utt2, rate = len_/len2)
                    
                   if is_triplet:
                   
                      utt3 = librosa.effects.time_stretch(utt2, rate = len_/len3)
                 
                       
                 
                 
             if not is_triplet:
                 
                
                return {'0': torch.tensor(utt), '1': torch.tensor(utt2)}
                
             else:
                
                return {'0': torch.tensor(utt), '1': torch.tensor(utt2), '2': torch.tensor(utt3)}
          else:
          
              utt = self.X[item]   
            
             
              utt = librosa.load(utt, sr = 16000, mono = True)
              
              clip = 3*16000
              
              if len(utt) > clip:
              
                  slices = len(utt)//clip
                  
                  utt = random.sample([utt[i*clip: (i+1)*clip] for i in slices \
                  
                                       if len(utt[i*clip: (i+1)*clip]) !< 2*16000 ], 1)[0]
                                       
                                       
              shift = random.sample(np.arange(3,9),1)[0]                     
              utt2 = librosa.effects.pitch_shift(utt,\
                                       sr =16000, n_steps = shift)  
                                       
                                       
              utt2 = librosa.effects.time_stretch(utt,  rate = 1.2)
                  
                  
              return {'0': torch.tensor(utt), '1': torch.tensor(utt2)}
              
              
              
              
              
              
              
              

def DATALOADER(dataset, is_dict= True, is_triplet= False, single = False, same_length= False, BATCH_SIZE = 4, shuffle = [True, True]):


   #dataset = load('DATASET_STFT')

   train, test  = DATA(dataset,same_length ,is_triplet, is_dict, 0), \
   
                    DATA(dataset,same_length ,is_triplet, is_dict,1)
                    


   dataset_size = len(test)
   print(dataset_size)
   iter_epoch = int(dataset_size/BATCH_SIZE)
   
   if is_triplet:
   
      collate_func = collate_function2
      
   else:
      collate_func = collate_function
      
   if single:
   
       collate_func = collate_function3   

   train_dataloader = data.DataLoader(train, \
               collate_fn = collate_func,  batch_size = BATCH_SIZE, shuffle = shuffle[0] )

   test_dataloader = data.DataLoader(test, \
          collate_fn= collate_func,  batch_size = BATCH_SIZE, shuffle = shuffle[1] )

   return train_dataloader, test_dataloader, iter_epoch              
              
              
              

def train_function(dataloader,is_triplet, single,model,optimizer,step_counter,epochs,scheduler = None, device='cuda', debug = 0):

       model.train()
       for bi, d in enumerate(dataloader):

           if bi%20 == 0:
                  print(f'TRAIN_BATCH_ID = {bi},')


           
           x1 = d[0]#.to(device)

           if not single:
             x2 = d[1]#.to(device)
          
           if is_triplet:
           
             x3 = d[2]
           
           #print(x1[0].shape, x2[0].shape)

           optimizer.zero_grad()

           if single:

             loss, logs =  model(x1) 
             
           else:
             loss, logs = model(x1,x2, step_counter)
               
           if is_triplet:
           
             loss , logs = model(x1,x2,x3)
             

           loss.backward()
           optimizer.step()

         
           if bi%20 == 0:
                  print(f'{logs}')

           step_counter += 1
           if scheduler != None:
                  scheduler.step()




           if debug ==1 :

                        break


       return  step_counter 



def eval_function(dataloader,is_triplet, single, model,epochs,  device= 'cuda',debug = 0):

       loss_tracker = []

       model.eval()
       for bi, d in enumerate(dataloader):
           if bi%20 == 0:
                  print(f'VALID_BATCH_ID = {bi},')
           logs = {}
           
           x1 = d[0]#.to(device)

           if not single:
             x2 = d[1]#.to(device)
          
           if is_triplet:
           
             x3 = d[2]

           if single:

             loss, logs =  model(x1) 
             
           else:
             loss, logs = model(x1,x2)
               
           if is_triplet:
           
             loss , logs = model(x1,x2,x3) 

           loss = loss.detach().cpu().numpy()


           loss_tracker.append(loss)

           if bi%20 == 0:
                  print(f' {logs }')






           if debug ==1 :
            
                        break




       return sum(loss_tracker)              
                                                      


def Trainer2(model, optimizer,  is_dict= True, is_triplet= False, single = False, same_length= False,  epoch_start = 0,num_steps = 5000,scheduler = None, EPOCHS=100, patience= 5, name = None, device = 'cuda',debug = 0, batch_size = 4, ):

      train_dataloader, test_dataloader, num_iter = DATALOADER(BATCH_SIZE = batch_size )
      cnt = 0

      step_counter= epoch_start * num_iter
      print('Start Step= ', step_counter)


      if epoch_start > 0 :
           print(f"|||||||||EVALUATING MODEL||||||||||")

           loss_ =  eval_function(test_dataloader, is_triplet, single,\
                                         model,epoch_start, device,debug)
           print(f'\nModel Loss = {loss_} \n')
           loss_book = [loss_]
      else:
      #     clustering(train_dataloader, model, debug)

           loss_book = [np.inf]



      for epochs in range(EPOCHS):
         if epochs >= epoch_start:



            print(f"\n||||||EPOCH = {epochs}||||||\n<<<~TRAINING~>>>\n")


            step_counter = train_function(train_dataloader, is_triplet, single, model,\
                          optimizer, step_counter, epochs, scheduler, device,debug)
            print(f"\n||||||EPOCH = {epochs}||||||\n<<<~VALIDATING~>>>\n")
            loss = eval_function(test_dataloader, is_triplet, single,\
                                         model,epochs, device,debug)



            loss = loss/num_iter
            loss_book.append(loss)


            print(f"\n||||||EPOCH = {epochs}, Final Validation Loss = {loss} ||||||\n")


            if debug==1:

                      break
            else:

              if loss < min(loss_book[:-1]):
                   print(f'\n|||||SAVING MODEL at epoch {epochs} |||||\n')
                   save_weights(model, f'best_{epochs}' , name= name)

                   cnt = 0


              else:
                   cnt += 1

              if epochs % 5 == 0 :
                    print(f'\n|||||SAVING MODEL at epoch {epochs} |||||\n')
                    save_weights(model, f'{epochs}', name = name)
              if cnt == patience:

                    cnt = 0
                    print(f"\n|||||Training DONE|||||\n")
                    break


      return loss_book

                                       
             
             
          
    
    
    
    
        


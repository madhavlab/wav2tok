
from training_function_library import Trainer 

from wav2tok import wav2tok

from new_function_library import load, save , get_linear_schedule_with_warmup

import torch 

from sklearn.model_selection import train_test_split



'''data = load('NPTEL_dataset')

### data in the form of a list of audios => [audio_dir1, audio_dir2, .....]

audios = ['../Dataset/nptel-pure-set/nptel-pure/wav/'+ keys for keys in data.keys()]

#audio_dict = {i : [audios[i]]*10 for i in range(len(audios))}

#train, test = audio_dict, audio_dict


train, test = train_test_split(audios, test_size = 0.6)

valid , test = train_test_split(test, test_size = 0.5)
save([train, valid], 'audios')

save(test, 'test_audios')'''


if __name__ == '__main__':
 
 

    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", type = int,nargs='?', default = 0, required='True' )
   
    parser.add_argument("--dataset", type = str ,nargs='?', default = None, required='True' )

    parser.add_argument("--sample_subdataset", type = bool ,nargs='?', default = False )
    
    parser.add_argument("--subdata_split", type = float ,nargs='?', default = 0.1 )
       
    parser.add_argument("--sr", type = int ,nargs='?', default = 16000, required='True' )

    parser.add_argument("--clip", type = bool ,nargs='?', default = False )

    parser.add_argument("--clip_duration", type = float ,nargs='?', default = 3.0 )
  
    parser.add_argument("--cluster_split", type = float ,nargs='?', default = 1.0, required='True' )
    
    parser.add_argument("--iter_clust", type = int ,nargs='?', default = 1.0, required='True' )

    parser.add_argument("--input_dim", type = int , nargs = '?' , default = 39, required = 'True') 
    
    parser.add_argument("--emb_dim",  type = int , nargs = '?' , default = 256, required = 'True')
    
    parser.add_argument("--num_tokens", type = int , nargs = '?' , default = 50, required = 'True')

    parser.add_argument("--num_layers", type = int , nargs = '?' , default = 2)

    parser.add_argument("--use_cosine", type = bool ,nargs='?', default = False )

    parser.add_argument("--temp", type = float ,nargs='?', default = 0.1 )

    parser.add_argument("--alpha", type = float ,nargs='?', default = 0.1 )
    
    parser.add_argument("--beta", type = float ,nargs='?', default = 0.1 )

    parser.add_argument("--use_transformer", type = bool ,nargs='?', default = False )

    parser.add_argument("--device", type = str,nargs='?', default ='cuda:0' )

    parser.add_argument("--batch_size", help = 'Batch Size', type = int,nargs='?', default = 4, required='True')

    parser.add_argument("--save_dir", type = str,nargs='?', default ='TrialTok', required= 'True' )
    
    parser.add_argument("--load_dir", type = str,nargs='?', default = None)
    
    parser.add_argument("--load_model_epochid", type = str,nargs='?', default = None)

    parser.add_argument("--best_model",type = str,nargs='?', default = True)

    parser.add_argument("--EPOCHS", type = int,nargs='?', default = 50 )
    
    parser.add_argument("--patience", type = int,nargs='?', default = 3 )
    
    parser.add_argument("--learning_rate", type = float,nargs='?', default = 2e-3 )
    
    parser.add_argument("--use_scheduler", type = int,nargs='?', default = True)
    
    parser.add_argument("--train_steps", type = int , nargs = '?' , default = None)

    parser.add_argument("--warmup",type = float, nargs = '?', default = 0.08)
 
    parser.add_argument("--epoch_start", type = int , nargs = '?' , default = 0)




    args = parser.parse_args()



debug = 0
#print(data)
dataset= 'audios' ###### wav2tok/bin/bird_audio.bin == [X_train, X_test]


model = wav2tok(39,256, dataset = dataset, mfcc = True, debug = debug ).cuda()



batch_size = 16

D = load('audios')
dataset_length = len(D)

EPOCHS = 10

train_steps = EPOCHS* dataset_length//batch_size

warmup = 0.08


learning_rate = 2e-3
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, betas = (0.9, 0.98), eps = 1e-6, weight_decay = 1e-2 )
 
scheduler = get_linear_schedule_with_warmup(optimizer,
                     num_warmup_steps = int(train_steps* warmup), \
                   num_training_steps = train_steps  )


Trainer(model= model, optimizer = optimizer, dataset =dataset, is_dict =False, apply_augmentation = True, scheduler = scheduler, clip = True , clip_duration = 3 , sr =16000, EPOCHS = EPOCHS, autosave = 2, name = 'Trialtok', debug = debug, batch_size = batch_size )

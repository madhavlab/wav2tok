
from training_function_library import Trainer 

from wav2tok import wav2tok

from new_function_library import load, save , get_linear_schedule_with_warmup
import argparse
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

    parser.add_argument("--debug", type = int,nargs='?', default = 0 )
   
    parser.add_argument("--dataset", type = str ,nargs='?', default = None, required='True' )

    parser.add_argument("--sample_subdataset", type = bool ,nargs='?', default = False )
    
    parser.add_argument("--subdata_split", type = float ,nargs='?', default = 0.1 )
       
    parser.add_argument("--sr", type = int ,nargs='?', default = 16000, required='True' )

    parser.add_argument("--clip", type = bool ,nargs='?', default = False )

    parser.add_argument("--clip_duration", type = float ,nargs='?', default = 3.0 )
    
    parser.add_argument("--mfcc", type = bool ,nargs='?', default = False)
    
    parser.add_argument("--is_dict", type = bool ,nargs='?', default = False, required = True)

    
    parser.add_argument("--is_triplet", type = bool ,nargs='?', default = False )

    parser.add_argument("--is_single", type = bool ,nargs='?', default = False )
    
    parser.add_argument("--same_length", type = bool ,nargs='?', default = False )

    parser.add_argument("--apply_augmentation", type = bool ,nargs='?', default = False )

    parser.add_argument("--autosave_epoch", type = int ,nargs='?', default = 5 )
  
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

    parser.add_argument("--name", type = str,nargs='?', default ='TrialTok', required= 'True' )
    
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
   
    D = load(args.dataset)
    dataset_length = len(D)


    model = wav2tok( args.input_dim , args.emb_dim, alpha = args.alpha, beta = args.beta,temp = args.temp , is_dict = args.is_dict,\
                         dataset= args.dataset , iter_clust = args.iter_clust, cluster_split = args.cluster_split, \
                          use_cosine = args.use_cosine,  use_transformer = args.use_transformer, \
                                       num_tokens= args.num_tokens, num_layers= args.num_layers, mfcc = args.mfcc, sr = args.sr, \
                            clip_duration = args.clip_duration, device = args.device, debug = args.debug ).to(args.device)

    
    if args.load_dir is not None: 
     
        if args.best_model:
           load_weights(model,'best_'+ args.load_model_epochid, args.load_dir)
        
        else:
           load_weights(model, args.load_model_epochid, args.load_dir)
    
    

    if args.train_steps is None:
         train_steps = args.EPOCHS * dataset_length // args.batch_size
    else:
         train_steps = args.train_steps

    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, betas = (0.9, 0.98), eps = 1e-6, weight_decay = 1e-2 )
 
    if args.use_scheduler:
         scheduler = get_linear_schedule_with_warmup(optimizer,
                     num_warmup_steps = int(train_steps* args.warmup), \
                   num_training_steps = train_steps  )
    else:
         scheduler =None







    Trainer(model = model, optimizer = optimizer, scheduler = args.scheduler, dataset = args.dataset, sample_subdataset = args.sample_subdataset, \
                subdata_split = args.subdata_split, is_dict= args.is_dict,  is_triplet= args.is_triplet, is_single = args.is_single, \
                same_length= args.same_length, apply_augmentation = args.apply_augmentation, clip = args.clip, clip_duration = args.clip_duration, \
                sr = args.sr, epoch_start = args.epoch_start , EPOCHS=args.EPOCHS, autosave= args.autosave_epoch, patience= args.patience,\
                name = args.name, device = args.device, debug = args.debug, batch_size = args.batch_size)


from training_function_library import Trainer 

from wav2tok import wav2tok

from new_function_library import load, save , get_linear_schedule_with_warmup

import torch 

from sklearn.model_selection import train_test_split



data = load('NPTEL_dataset')

### data in the form of a list of audios => [audio_dir1, audio_dir2, .....]

audios = ['../Dataset/nptel-pure-set/nptel-pure/wav/'+ keys for keys in data.keys()]

audio_dict = {i : [audios[i]]*10 for i in range(len(audios))}

train, test = audio_dict, audio_dict


#train, test = train_test_split(audios, test_size = 0.1)
save([train, test], 'audios')

#print(data)
dataset= 'audios' ###### wav2tok/bin/bird_audio.bin == [X_train, X_test]


model = wav2tok(39,256, dataset = dataset, mfcc = True, debug = 1 ).cuda()

batch_size = 1

D = load('audios')
dataset_length = len(D)

EPOCHS = 2

train_steps = 2* dataset_length//batch_size

warmup = 0.08


learning_rate = 2e-3
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, betas = (0.9, 0.98), eps = 1e-6, weight_decay = 1e-2 )
 
scheduler = get_linear_schedule_with_warmup(optimizer,
                     num_warmup_steps = int(train_steps* warmup), \
                   num_training_steps = train_steps  )


Trainer(model= model, optimizer = optimizer, dataset =dataset, is_dict =True, apply_augmentation = True, scheduler = scheduler, clip = True , clip_duration = 1 , sr =16000, EPOCHS = 2, autosave = 1, name = 'Trialtok', debug = 1, batch_size = 1 )

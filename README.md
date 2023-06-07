# wav2tok: Deep Sequence Tokenizer for Audio Retrieval
Codes for Reproducibility

Paper link: https://openreview.net/forum?id=v8Mi8KU6056

Citation (bibtex): 

    @inproceedings{banerjee2023wav2tok,
     title={wav2tok: Deep Sequence Tokenizer for Audio Retrieval},
     author={Banerjee, Adhiraj and Arora, Vipul},
     booktitle={The Eleventh International Conference on Learning Representations},
     year={2023}
     }


# Repository Structure:


wav2tok/Src 


make 3 more folders bin, weights, Dataset

wav2tok/Src

       /bin 
       
       /weights
       
       /Dataset




#   Training wav2tok 


Keep dataset in wav2tok/Dataset



Make a list of data splits and save as .bin file wav2tok/bin

    audios.bin == [X_train , X_test]

    X_train, X_test -> lists of audio paths

                       [audio path 1, audio path 2, ...]

                     
   ##                    OR



Make a list of data dictionaries and save as .bin file to wav2tok/bin 

      audios.bin == [X_train , X_test]

      X_train, X_test -> dictionaries of audio

      Keys -> Classes or labels (song_id1, song_id2)

      Values -> List of audio paths (10 utterances for song_id1)


                {class 1 : list of audio , class 2 : list of audio ...}


# Code for Training wav2tok: 

We have a dedicated function for training a wav2tok model.

                     wav2tok/Src/train.py

Functions used in wav2tok/Src/train.py:

     wav2tok() from wav2tok/Src/wav2tok.py

    Trainer() from wav2tok/Src/training_function_library.py


To train a wav2tok model just run in command prompt,

     python3 train.py --args1 arg_value1 --args2 arg_value2 
   
   
# Arguments to pass: 
       



# Details of Args for class wav2tok





    --debug -> int, 1 for debug mode, 0 for work mode  

    --use_transformer -> Boolean (Default: False), if you want to use a transformer network as encoder ,
                                             but you have to set the args in wav2tok/Src/wav2tok.py
                                             in class TransformerEncoder and TransformerSentenceEncoderLayer

                                             We use BiLSTM encoder, you can tweak parameters 
                                                            in wav2tok/Src/wav2tok.py class Emb
       




    --input_dim -> input Feature dim (STFT dim) 

    --emb_dim -> Embedding dim (encoder output dim)

    --num_tokens -> number of tokens to use for tokenization 

    --num_layers -> number of layers to use for BiLSTM model (no effect if you want to use Transformer) 
              (Default: 2)





    --device -> str (Default: 'cuda'), GPU device name



    --dataset -> str (Default: None),Dataset name for clustering ('audios') / takes the training spilt for clustering

            

          
    --cluster_split -> percentage of training data to use for clustering (data is sampled randomly)
                 (Default: 1.0)


    --iter_clust -> number of training steps before each clustering session
              (Default: 500 training steps)


    --use_cosine -> use cosine similarity in matching task instead of parameterized similarity score
              (Default: False)              



    --temp -> temperature for the logits used in cross-entropy calculation
        (Default: 0.1)


    --alpha , --beta -> positive constants in likelihood loss
                (Default: 0.01,0.01)



#  Details of Args for Trainer(...) function 


    --debug -> int, 1 for debug mode, 0 for work mode  

    
   
    --dataset -> str, Dataset filename (dataset: {filename}.bin)





    --model -> Class wav2tok , model instance

    --optimizer -> Class torch optimizer, optimizer instance
 
    --scheduler -> Class torch scheduler, learning rate scheduler

    --is_dict -> Boolean (Default: False), if Dataset is a dictionary or list 
    
    
    --sample_subdataset -> Boolena (Default: False), sample random subsets of data for training on Large datasets 
                           Works only if --is_dict == False

    --subdata_split -> float (Default: 0.1), How big of a portion are the subdatasets in comparison to the large dataset
    
    --is_triplet -> Boolean (Default: False), if you want to train with Batches of Triplets (anchor, positive, negative)
                    
    --is_single -> Boolean (Default: False), if you want to train with batches of audio (anchor)
        
  
    ##########    Default Training is done with pairs of audio (anchor, positive) ##############


    --same_length -> Boolean (Default: False), if you want to time stretch audios in each batch of (anchor) or (anchor. positive), (anchor, positive, negative) to same length  

    --apply_augmentation -> Boolean (Default: False), works if is_dict == True, apply augmentations to pairs sampled from dictionary === (anchor, positive), apply augmentation to positive

    --clip -> Boolean (Default: False), works if is_dict = False, if you want to clip the to some duration
     
    --clip_duration -> float (Default: 3), clip audio to {clip_duration} seconds

    --sr -> int (Default: 16000), sampling rate of audio

    --batch_size -> int (Default: 4), Training batch size

    --EPOCHS -> int (Default: 100), Number of full data passes 

    --autosave -> int (Default: 5), autosave model parameters in {autosave} number of epochs

    --patience -> int (Default: 5), stop training if evaluation metric doesn't increase for {patience} number of epochs

    --name -> str (Default: 'TrialTok' ), Model parameters save filename 

    --epoch_start -> int (Default: 0), To start training at {epoch_start} epoch.

    --device -> str (Default: 'cuda'), GPU device name
#Details on Args for optimizer, learning rate scheduler, weight saving and loading: 
     
      --learning_rate ->   float (Default: 2e-3), Learning rate for Training
      
      --train_steps -> int (Default: None, Calculated as EPOCHS* dataset_length //batch_size), number of training steps

      --warmup -> float (Default: 0.08), Percentage of training steps to be used for warm up 

      --save_dir -> str (Default: Trialtok), Name of model for weight save file

      --load_dir -> str (Default: None), Model name to load
      --load_model_epochid -> int (Default: None), Epoch id to load 
      
      
# Brief on the functions present in class wav2tok:


     forward -> input: seq1 , seq2, training_steps

           output: loss with gradients, logs



    cluster -> input: dataset name -> string ('bird_audio')

           output: Performs clustering and 
                   sets the token classifier codebook



     get_feats -> input: audio-> wav , mfcc -> Boolean 

             output: MFCCs if MFCC == True else STFT matrix 
                     (you can the parameters for extraction of features 
                      manually inside the code segment )



    get_embs -> input: audio -> STFT or MFCC

            output: numpy array of Embeddings


    initialize_classifier -> input ->  Codebook of token representations, 
                                   shape: (number of tokens, Embedding dim)
                         output -> sets token classifier codebook as input





    ctc_loss_cal -> input: logits of shape (Time, classes), token sequence

                output: CTC loss or likelihood loss


    gen_prototype -> input: Concatenated sequences of representations {Z, Z'},
                        Concatenated sequences of tokens {T,T'},
                        Unique tokens in concatenated sequence {T,T'}
            
                 output: Dictionary of Prototypes {token: prototype or average representation in {Z,Z'}  mapping to token }



    matching_loss -> input: Dictionary of Prototypes
 
                 output: contrastive loss 


  
    inter_dict_weights -> calculates distances from codebook representations
                      Helper function to matching_loss


 
    

#  Weight Saving and loading 


The Trainer function saves the best weights as well as weights every 5 (default) epochs 

Uses load_Weights and save_Weights functions in wav2tok/Src/new_function_library.py




    save_weights -> input: model instance, epoch_id, name

                output: save weights in wav2tok/weights/{name}_{epoch}.bin



    load_Weights -> input: model instance, epoch_id, name 

                output: load weight to model




#  Args to pass to wav2tok/Src/train.py for different cases of audio dataset 


#  Case 1:



wav2tok/bin/audio.bin == [X_train, X_test]

X_train, X_test -> dictionaries 
                   {class 1 : list of audio , class 2 : list of audio ...}

   

    python3 train.py --dataset audios --is_dict True  --is_pairwise True --apply_augmentation True



    apply_augmentation = True, if you want to sample another sequence of same class 
                           and apply augmentation to it



                     False , if you want to only sample another sequence of same class





#  Case 2:

 


wav2tok/bin/audio.bin == [X_train, X_test]

X_train, X_test -> list of audio paths [audio path 1, audio path 2, ...]

     python3 train.py --dataset audios --is_dict False  --is_pairwise True 


    apply_augmentation = doesn't matter similar sequence generated via audio augmentations



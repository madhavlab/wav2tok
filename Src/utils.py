import librosa
import numpy as np
import pdb
import string
from Levenshtein import distance
from sklearn.preprocessing import MinMaxScaler
import torch
from typing import Callable, Dict, List, Optional


import torch.nn.functional as F
def wav2feat(wavfile):
    '''
    Input: audio wav file name
    Output: Magnitude spectrogram
    '''
    x, Fs = librosa.load(wavfile, sr=44100, mono=True) 
    hop = int(0.01 * Fs) # 10ms
    win = int(0.02 * Fs) # 20ms
    X = librosa.stft(x, n_fft=1024, hop_length=hop, win_length=win, window='hann', center=True, pad_mode='reflect')
    X = np.abs(X)
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    
    return X

def wavs2feat(wavfiles):
    '''
    Concatenate the audio files listed in wavfiles
    Input: list of audio wav file names
    Output: Magnitude spectrogram of concatenated wav
    '''
    x = []
    for wf in wavfiles:
        x1, Fs = librosa.load(wf, sr=44100, mono=True)
        x.append(x1)
    x = np.hstack(x)
    hop = int(0.01 * Fs) # 10ms
    win = int(0.02 * Fs) # 20ms
    X = librosa.stft(x, n_fft=1024, hop_length=hop, win_length=win, window='hann', center=True, pad_mode='reflect')


    
    return np.abs(X)

def read_csv(filename):
    id_label = {}
    with open(filename,'r') as fid:
        for line in fid: # '176787-5-0-27.wav,engine_idling\n'
            tokens = line.strip().split(',') # ['176787-5-0-27.wav', 'engine_idling']
            id_label[tokens[0]] = tokens[1]
    return id_label

def editDistance(gt, est):
    '''both are lists of labels
    E.g. gt is "dog_bark-street_music-engine_idling"
    E.g. est is "street_music-engine_idling"
    '''
    gt = [str(i) for i in gt]
    est = [str(i) for i in est]
    gttokens = gt #.split('-')
    esttokens = est #.split('-')
    # Map token to char
    tokenset = list(set(gttokens+esttokens)) # ['dog_bark', 'siren', 'street_music', 'engine_idling']
    token_char = {}
    for i in range(len(tokenset)):
        token_char[tokenset[i]] = str(i)  # {'dog_bark': 'A', 'siren': 'B', 'street_music': 'C', 'engine_idling': 'D'}
    # convert gt and est to strings
    gtstr = [token_char[t] for t in gttokens]
    gtstr = ''.join(gtstr)  # 'BCA'
    eststr = [token_char[t] for t in esttokens]
    eststr = ''.join(eststr)  # 
    # Compare
    editdist = distance(gtstr, eststr) # 1
    score = 1 - editdist/len(gtstr)
    return editdist, score

def evals(gtcsv, estcsv, taskid):
    gt_id_label = read_csv(gtcsv)
    est_id_label = read_csv(estcsv)
    score = 0
    for id in est_id_label:
        if taskid==1:
            if est_id_label[id] == gt_id_label[id]:
                score += 1
        elif taskid==2:
            _, ss = editDistance(gt_id_label[id], est_id_label[id])
            score += ss
        else:
            pdb.set_trace()
            assert False, ["taskid not correct; it is", taskid]
    avgScore = score/len(est_id_label)
    return avgScore




def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


def get_available_activation_fns() -> List:
    return [
        "relu",
        "gelu",
        "gelu_fast",  # deprecated
        "gelu_accurate",
        "tanh",
        "linear",
    ]


def get_activation_fn(activation: str) -> Callable:
    """ Returns the activation function corresponding to `activation` """
    from modules import gelu, gelu_accurate

    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_fast":
        deprecation_warning(
            "--activation-fn=gelu_fast has been renamed to gelu_accurate"
        )
        return gelu_accurate
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))




def item(tensor):
    if hasattr(tensor, "item"):
        return tensor.item()
    if hasattr(tensor, "__getitem__"):
        return tensor[0]
    return tensor





'''if __name__=="__main__":
    wavs = ['../shared_train/audio_train/180937-7-3-27.wav']
    stft = wavs2feat(wavs)
    print(stft.shape)
     
    wavfiles = ['../shared_train/audio_train/180937-7-3-27.wav','../shared_train/audio_train/180937-7-3-27.wav']
    X = wavs2feat(wavfiles)
    eval('test_task1/labels.csv', 'test_task1/est.csv', 1)
    editDistance("dog_bark-street_music-engine_idling",
         "siren-street_music-engine_idling")'''

    


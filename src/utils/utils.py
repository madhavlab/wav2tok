import torch
from utils.audio_handler import Audio ####changed from utils.audio_handler to audio_handler due to import issue in baselines/HuBERT/eval_jaccard.ipynb
import math
import torch.nn.functional as F
import os
import numpy as np


def get_utterance(metadata: dict, fs: int)-> torch.FloatTensor:
    audioreader = Audio()
    Audio.sample_rate = fs
    utterance_data = audioreader.read(metadata["filename"])[int(metadata["start"]*fs):int(metadata["end"]*fs)]
    return utterance_data


def get_fixed_length_utterance_pair(utterance_pair_metdata, sample_rate=16000, audio_length=1.0, apply_preemphasis=True, context_padding=True):
    audioreader = Audio()
    Audio.sample_rate = sample_rate
    audio_length = int(sample_rate*audio_length)

    utt1, utt2 = utterance_pair_metdata

    # read audio tracks containing utterances
    audio1_data = audioreader.read(utt1["filename"], apply_preemphasis=apply_preemphasis)
    audio2_data = audioreader.read(utt2["filename"], apply_preemphasis=apply_preemphasis)

    # get padding lengths
    utt1_len, utt2_len = int(utt1["end"] *sample_rate) - int(utt1["start"] *sample_rate), int(utt2["end"] *sample_rate) - int(utt2["start"] *sample_rate) 
    utt1_pad_len, utt2_pad_len = (audio_length - utt1_len)/2, (audio_length - utt2_len)/2

    if context_padding:
        # get zero padding lengths --> cases when utterances are near track boundaries | add padding
        utt1_zero_pad_l, utt1_zero_pad_r = max(0,math.floor(utt1_pad_len)-(int(utt1["start"]*sample_rate))), max(0,math.ceil(utt1_pad_len)-(len(audio1_data)-int(utt1["end"]*sample_rate)))
        utt1_data = audio1_data[max(0,(int(utt1["start"]*sample_rate))-math.floor(utt1_pad_len)) : min(len(audio1_data), int(utt1["end"]*sample_rate)+math.ceil(utt1_pad_len))]
        utt1_data = F.pad(utt1_data, (utt1_zero_pad_l, utt1_zero_pad_r))

        utt2_zero_pad_l, utt2_zero_pad_r = max(0,math.floor(utt2_pad_len)-(int(utt2["start"]*sample_rate))), max(0,math.ceil(utt2_pad_len)-(len(audio2_data)-int(utt2["end"]*sample_rate)))
        utt2_data = audio2_data[max(0,(int(utt2["start"]*sample_rate))-math.floor(utt2_pad_len)) : min(len(audio2_data), int(utt2["end"]*sample_rate)+math.ceil(utt2_pad_len))]
        utt2_data = F.pad(utt2_data, (utt2_zero_pad_l, utt2_zero_pad_r))
    else:
        utt1_data = audio1_data[(int(utt1["start"]*sample_rate)): (int(utt1["end"]*sample_rate))]
        utt2_data = audio2_data[(int(utt2["start"]*sample_rate)): (int(utt2["end"]*sample_rate))]
        utt1_data = F.pad(utt1_data, (math.floor(utt1_pad_len), math.ceil(utt1_pad_len)))
        utt2_data = F.pad(utt2_data, (math.floor(utt2_pad_len), math.ceil(utt2_pad_len)))

    assert len(utt2_data) == len(utt1_data) == audio_length, f"utterance lengths with added context mismatch {len(utt2_data), len(utt1_data)}. It must be equal to {audio_length}"

    utt1 = {"data":utt1_data, "pad_length":utt1_pad_len}
    utt2 = {"data":utt2_data, "pad_length":utt2_pad_len}
    return utt1, utt2


def get_pad_mask(pad_len:int, token_interval=160, total_frames=101):
    pad_frames_len = math.floor((pad_len/token_interval))
    mask = torch.zeros(total_frames, dtype=torch.int8)
    if pad_frames_len > 0:
        mask[:pad_frames_len] = 1
        mask[-pad_frames_len:] = 1
    return mask



class MemoryMappedArray:
    def __init__(self, filename, dtype, B, W, L):
        self.filename = filename
        self.dtype = dtype
        self.B = B
        self.W = W
        self.L = L
        if os.path.exists(os.path.dirname(filename)) is False:
            os.makedirs(os.path.dirname(filename))
        self.array = np.memmap(filename, dtype=dtype, mode='w+', shape=(B,W,L))
        self.current_size = 0
    
    def append(self, data):
        new_frames = data.shape[0] 
        required_frames = self.current_size + new_frames
        
        # Resize if necessary
        if required_frames > self.B:
            # print("increasing shape")
            self.B = 2*self.B
            # Close the current memmap to release the file
            del self.array
            
            # Extend the file by truncating to the new size
            with open(self.filename, 'ab') as f:
                f.truncate(np.array(self.B * self.W * self.L * np.dtype(self.dtype).itemsize))
            
            # Reopen with the updated shape
            self.array = np.memmap(self.filename, dtype=self.dtype, mode='r+', shape=(self.B,self.W,self.L))
        
        # Write new data to the expanded portion
        self.array[self.current_size:required_frames] = data
        self. array.flush()  # Save changes to disk
        self.current_size = required_frames
        return self.array[:self.current_size]

    def getdata(self):
        return self.array[:self.current_size]

    @property
    def size(self):
        return self.current_size
        


def generate_query(word_alignments, model=None, lang=None, hop=0.10, audio_parent_path="/home/adhirajb/TSS/w2t_std/DATA", only_metadata=False, verbose=False):
    if lang is None:
        lang = np.random.choice(list(word_alignments.keys()))
    
    while True:
        words = list(word_alignments.keys())
        #print(words)
        query_word = np.random.choice(words)

        if len(word_alignments[query_word]) > 3 and len(word_alignments[query_word]) < 15 and len(query_word)>5:
            break
    
    if verbose:
        print(f"language: {lang}\nword: {query_word}\ntotal occurences: {len(word_alignments[query_word])}")
    
    if only_metadata:
        return lang, query_word, len(word_alignments[query_word])
    
    else:
        assert model is not None, "passed model is None"
        files_metadata_containing_query = word_alignments[query_word]
        #print(files_metadata_containing_query)
        query_dbase = {}
        for filename, timebd in files_metadata_containing_query.items():
            try:
                filepath = os.path.join(audio_parent_path, filename)
                query_metadata = {"filename": filepath, "start": timebd[0], "end": timebd[1]}
                #print("metadata", query_metadata)
                query_data, _ = get_fixed_length_utterance_pair([query_metadata, query_metadata], context_padding=True)
                #print("query_data", query_data)
                audio_tokens = model.extract_tokens(query_data["data"], hop=hop,chunksz = 16, gpu_index="0")
                z = None
                query_dbase[filepath] = [audio_tokens, z, query_data]
            except Exception as e:
                print(e)
            continue
        # queries_fname = ["Kathbath"+fname.split("Kathbath",1)[-1] for fname in list(query_dbase.keys())]
        queries_fname = list(query_dbase.keys())
        return queries_fname, query_dbase, lang, query_word, len(word_alignments[query_word])


# for meity
def extract_query_tokens(recored_query_path, model, search_api, fs=16000):
    Audio.sample_rate=fs
    audioreader = Audio()
    audiodata = audioreader.read(recored_query_path)
    z, audio_tokens = model.extract_tokens(audiodata, hop=0.10, chunksz=16, gpu_index=0)
    audio_tfidf = search_api.get_tfidf_repr(audio_tokens)
    print("Preprocessing done!")
    return audio_tfidf, audio_tokens[0], z[0], audiodata
import os
import sys
#sys.path.append("../")
import os
import glob
import pickle
import torch
import faiss
import numpy as np
from tqdm import tqdm
from typing import List, Optional
from Levenshtein import ratio
from tslearn.metrics import  dtw_path_from_metric

from utils import Audio, MemoryMappedArray, generate_query
from train import AudioTokenizer
from npy_append_array import NpyAppendArray
from retrieval import STD




codebook_size = 512
split="TRAIN"
api = STD(codebook_size=codebook_size,
          parent_dbase_dir= "/home/adhirajb/TSS/wav2tok/database/DEV2/", 
          model_checkpoint="/path/to/checkpoint.ckpt")

# ref_fnames= api.get_reference_fnames(audio_parent_path="/home/adhirajb/TSS/wav2tok/DATA/timit",
#                          lang="english",
#                          splits=["TRAIN"])

# api.build_database(hop=0.1) #, gpu_index=0)

# api.build_tfidf_dbase(lang="english")

# api.build_index(lang="english")

import torch.nn.functional as F
#initiate search setup
checkpoint_path="/path/to/checkpoint.ckpt"
model = AudioTokenizer.load_from_pretrained(checkpoint_path, gpu_index=0)
codebook = F.normalize(model.vq_layer.codebook.detach().cpu(), dim=-1)
api.initiate_search_setup()

# create queries for dummy purposes
alignment_path = "/home/adhirajb/TSS/wav2tok/DATA/timit/timit_word_alignment.pkl"
word_alignments = pickle.load(open(alignment_path, "rb"))
result_logs = []
for i in range(100):
    queries_fname, query_dbase, query_lang, query_word, query_counts = generate_query(word_alignments, model, lang="english")
    print(query_word)
    #print(query_dbase.keys())

    query_idx = 0
    query_tokens, query_z, query_data = query_dbase[list(query_dbase.keys())[query_idx]]
    query_tfidf = api.tfidf_repr(query_tokens, normalize=True)
    #print(query_tokens)
    query = [query_tfidf, query_tokens[0], query_z]
    #display(Audio(query_data["data"], rate=16000))


    results_log, matches = api.search(query=query,
                                    search_lang="english",
                                    queries_fname=queries_fname, 
                                    codebook=codebook,
                                    index_probe=20, 
                                    tfidf_topk=10000,
                                    jaccard_thresh=0.35,
                                    edit_dist_thresh=0.45,
                                    dtw_quantz_thresh=0.15,
                                    dtw_z_thresh=0.2,
                                    levels=3,
                                    verbose=False,
                                    get_stats=True
                                    )
    result_logs.append(results_log)


print(result_logs)


avg_res = np.array(result_logs).mean(0)
print(f"retrieved cands: {avg_res[0::3]}")
print(f"recall: {avg_res[1::3]}")
print(f"twv: {avg_res[2::3]}")
print("##########################################################")
# print(matches)

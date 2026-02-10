import os
import sys
sys.path.append("../")
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

from utils import Audio
from train import AudioTokenizer
from npy_append_array import NpyAppendArray


class STD:
    def __init__(self,
                codebook_size,
                parent_dbase_dir:str,
                # encoding_dims,
                # frame_length: int=101,
                sample_rate: int=16000,
                model_checkpoint: Optional[str]=None,
                reference_files_fname: Optional[str]="ref_filenames.csv",
                zdbase_fname: Optional[str]= "memmap_zdbase.dat",
                tokensdbase_fname: Optional[str]= "memmap_tokensdbase.dat",
                tfidfdbase_fname: Optional[str]= "memmap_tfidfdbase.dat",
                fileidx_cumsum_map_fname: Optional[str] = "fileidx_cumsum_map.pkl",
                fileidx_embidx_map_fname: Optional[str] = "fileidx_embidx_map.pkl",
                index_fname: Optional[str]="tfidf.index",
                lang_startendidx_fname: Optional[str] = "lang_start_end_embidx.pkl",
                ref_files_count_per_lang_fname: Optional[str] = "ref_files_count_per_lang.pkl"
                ):
        
        self.codebook_size = codebook_size
        self.parent_dbase_dir = parent_dbase_dir
        self.model_checkpoint = model_checkpoint
        self.reference_files_fname = reference_files_fname
        self.zdbase_fname = zdbase_fname
        self.tokensdbase_fname = tokensdbase_fname
        self.tfidfdbase_fname = tfidfdbase_fname
        self.fileidx_cumsum_map_fname = fileidx_cumsum_map_fname
        self.fileidx_embidx_map_fname = fileidx_embidx_map_fname
        self.index_fname = index_fname
        self.lang_startendidx_fname = lang_startendidx_fname
        self.ref_files_count_per_lang_fname = ref_files_count_per_lang_fname

        Audio.sample_rate = sample_rate
        self.audioreader = Audio()

        os.makedirs(self.parent_dbase_dir, exist_ok=True) 
        os.makedirs(os.path.join(self.parent_dbase_dir, "metadata"), exist_ok=True)
        os.makedirs(os.path.join(self.parent_dbase_dir, "database"), exist_ok=True)
        os.makedirs(os.path.join(self.parent_dbase_dir, "index"), exist_ok=True)

    @staticmethod
    def jaccard_similarity(tokens1, tokens2):
        # computes intersection over union

        if len(tokens1.shape) == 2:
            tokens1 = tokens1[0]
        if len(tokens2.shape) == 2:
            tokens2 = tokens2[0]

        if isinstance(tokens1, torch.Tensor) or isinstance(tokens1, np.ndarray):
            tokens1 = tokens1.tolist()
            tokens2 = tokens2.tolist()

        unique_tokens1 = set(tokens1)
        unique_tokens2 = set(tokens2)
        intersect = unique_tokens1.intersection(unique_tokens2)
        union = unique_tokens1.union(unique_tokens2)
        sim = len(intersect)/len(union)
        return sim 

    def get_reference_fnames(self,
                            audio_parent_path: str,
                            audio_ext: Optional[str]="WAV",
                            lang: Optional[str]="all_langs",
                            splits: Optional[List[str]]=None,
                            N: Optional[int]=None,
                            query_word: Optional[str]=None,
                            query_lang: Optional[str]=None,
                            word_alignments: Optional[str]=None) -> np.ndarray:            
        
        if splits is None:
            splits = ["**"]
        if lang == "all_langs":
            lang = "**"

        ref_fnames = []
        for split in splits:
            ref_fnames.extend(glob.glob(os.path.join(audio_parent_path,lang,  split,  "**" ,"**","*."+audio_ext)))

        if N is not None:
            ref_fnames = np.random.choice(ref_fnames, size=N, replace=False)

        # use it when running on a dummy database and to include the files corresponding query in the reference database 
        if query_word is not None:
            assert word_alignments is not None, "word_alignment path is not specified"
            assert query_lang is not None, "query language is not specified"
            audio_parent_path =  audio_parent_path.split("/DATA",1)[0] ####dataset dependent
            query_fnames = np.array([os.path.join(audio_parent_path, fname) for fname in word_alignments[query_lang][query_word].keys()])
            ref_fnames = np.append(ref_fnames, query_fnames)

        savepath = os.path.join(self.parent_dbase_dir,"metadata", self.reference_files_fname)
        np.savetxt(savepath, ref_fnames, fmt="%s")
        print(f"reference filenames saved at: {savepath}")
        return ref_fnames
    
    def build_database(self,
                        hop: Optional[float]=0.10):
        
        #builds embeddings(to check if need to store) and the tokens databases for all languages
        # its a big stack from where we can extract embeddings for lang-specific files
        assert self.model_checkpoint is not None, "model checkpoint path is None"
    
        # memmap_z = NpyAppendArray(filename=os.path.join(self.parent_dbase_dir, "database", self.zdbase_fname), delete_if_exists=True) # not going to use level 5 search so don't need it 
        memmap_tokens = NpyAppendArray(filename=os.path.join(self.parent_dbase_dir, "database", self.tokensdbase_fname), delete_if_exists=True)
        ref_fnames = np.loadtxt(os.path.join(self.parent_dbase_dir, "metadata", self.reference_files_fname), dtype=str)
        model = AudioTokenizer.load_from_pretrained(self.model_checkpoint, gpu_index=0)
        
        print("building database...")
        init_cumsum = 0
        cumsum=0
        fileidx=0
        fileidx_embidx_map = []
        fileidx_cumsum_map = []
        file_start_end_idx_per_lang = {}
        ref_files_count_per_lang = {}
        lang_index_ranges = {}
        failed_files_mask = np.ones(len(ref_fnames))
        for idx, fpath in enumerate(tqdm(ref_fnames)):
            try:
                #print(fpath)
                audio_data= self.audioreader.read(fpath)
                #print(audio_data.shape)
                audio_tokens = model.extract_tokens(audio_data, hop=hop, chunksz= 64, gpu_index="0")
                # memmap_z.append(z.numpy().astype(np.float16)) 
                #print(audio_tokens)
                memmap_tokens.append(audio_tokens.numpy().astype(np.int16))

                fileidx_embidx_map.extend([fileidx]*len(audio_tokens))
                fileidx+=1
                fileidx_cumsum_map.append(cumsum)
                cumsum += len(audio_tokens)


                # start and end indices in the memmap data for each language
                # here we assumed that filenames corresponding to each lang are stored consecutively, and thus,
                # the tokens/embeddings are also stored blockwise for each lang in a big array

                # to store the count of files per language:
                 
                lang = fpath.split("/timit")[1].split("/")[1]
                if lang not in file_start_end_idx_per_lang:
                    file_start_end_idx_per_lang[lang] = [idx, idx] # [start_file_idx end_file_idx]
                else:
                    file_start_end_idx_per_lang[lang][1] = idx
                    ref_files_count_per_lang[lang] = file_start_end_idx_per_lang[lang][1] - file_start_end_idx_per_lang[lang][0] + 1

                
                if lang not in lang_index_ranges:
                    lang_index_ranges[lang] = [init_cumsum, init_cumsum]
                else:
                    lang_index_ranges[lang][1] = cumsum
                    init_cumsum = cumsum

            except Exception as e:
                failed_files_mask[idx] = 0
                # print(f"error for filepath: {fpath}")
                print(e)
                continue
        # memmap_z.close()
        memmap_tokens.close()

        if failed_files_mask.sum() < len(ref_fnames):
            print(f"failed processing files: {len(ref_fnames)-failed_files_mask.sum()}")
            ref_fnames = ref_fnames[failed_files_mask.astype(bool)]
            print(f"updating reference filenames located at: {os.path.join(self.parent_dbase_dir, self.reference_files_fname)}")
            np.savetxt(os.path.join(self.parent_dbase_dir, "metadata", self.reference_files_fname), ref_fnames, fmt="%s")
        ref_files_count_per_lang["all_langs"] = len(ref_fnames)
        lang_index_ranges["all_langs"] = [0, cumsum]

        pickle.dump(ref_files_count_per_lang, open(os.path.join(self.parent_dbase_dir, "metadata", self.ref_files_count_per_lang_fname), "wb"))
        pickle.dump(lang_index_ranges, open(os.path.join(self.parent_dbase_dir, "metadata", self.lang_startendidx_fname), "wb"))
        pickle.dump(fileidx_cumsum_map, open(os.path.join(self.parent_dbase_dir, "metadata", self.fileidx_cumsum_map_fname), "wb"))
        pickle.dump(fileidx_embidx_map, open(os.path.join(self.parent_dbase_dir, "metadata", self.fileidx_embidx_map_fname), "wb"))
        print(f"database and metadata files saved at {self.parent_dbase_dir}")
        return

    def build_tfidf_dbase(self, lang="all_langs"):
        tokens_dbase = np.load(os.path.join(self.parent_dbase_dir, "database", self.tokensdbase_fname), mmap_mode="r")
        lang_index_ranges = pickle.load(open(os.path.join(self.parent_dbase_dir, "metadata", self.lang_startendidx_fname), "rb"))
        start, end = lang_index_ranges[lang]
        tokens_dbase = tokens_dbase[start:end]
        dbase_size = end-start

        print(f"building tf-idf for {lang} | database size: {dbase_size}")

        idf_count = {}
        for frame in tokens_dbase:
            unique_frame_tokens = set((frame.tolist()))
            for token in unique_frame_tokens:
                if token not in idf_count:
                    idf_count[token] = 0
                idf_count[token] += 1

        idf_count_save_path = os.path.join(self.parent_dbase_dir, "metadata", "idf_counts")
        os.makedirs(idf_count_save_path, exist_ok=True)
        pickle.dump(idf_count, open(os.path.join(idf_count_save_path, lang+"_idf_count.pkl"), "wb")) 

        # create tf-idf vector for each frame
        savedirpath = os.path.join(self.parent_dbase_dir, "database", "tfidf_dbases")
        os.makedirs(savedirpath, exist_ok=True) 
        tfidf_dbase = NpyAppendArray(filename=os.path.join(savedirpath, lang+"_"+self.tfidfdbase_fname), delete_if_exists=True)
        for frame_tokens in tqdm(tokens_dbase):
            tf_idf_vec = np.zeros(self.codebook_size, dtype=np.float16)
            u, c = np.unique(frame_tokens, return_counts=True)
            c_sum = c.sum() 
            for i, token in enumerate(u):
                tf_idf_vec[token] = (c[i]/c_sum) * np.log(dbase_size/idf_count[token])
            tfidf_dbase.append(tf_idf_vec[None,:])
        tfidf_dbase.close()
        return 
    
    def get_dbase_size(self, lang="all_langs"):
        lang_index_ranges = pickle.load(open(os.path.join(self.parent_dbase_dir, "metadata", self.lang_startendidx_fname), "rb"))
        start, end = lang_index_ranges[lang]
        dbase_size = end-start
        return dbase_size
    
    def build_index(self, lang:Optional[str]="all_langs", partitions: Optional[int]=100, subquantizers: Optional[int]=8, bits=8):
        tfidf_dbase = np.load(os.path.join(self.parent_dbase_dir, "database", "tfidf_dbases", lang+"_"+self.tfidfdbase_fname), mmap_mode="r")
        nlist = partitions
        m = subquantizers                             # number of subquantizers
        d = self.codebook_size
        b = bits                                      # b specifies that each sub-vector is encoded as b bits
        quantizer = faiss.IndexFlatIP(d) 
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, b)

        print(f"building index for lang: {lang}")
        print(f"total database size to index: {len(tfidf_dbase)}")                              
        index.train(tfidf_dbase)
        index.add(tfidf_dbase)
                
        savepath = os.path.join(self.parent_dbase_dir, "index", lang+"_"+self.index_fname)
        faiss.write_index(index, savepath)
        print(f"index saved at {savepath}")
        return index
    
    def initiate_search_setup(self):
        print("Retrieval system ready!")
        self.tokens_dbase = np.load(os.path.join(self.parent_dbase_dir, "database", self.tokensdbase_fname), mmap_mode="r")
        # self.z_dbase = np.load(os.path.join(self.parent_dbase_dir, "database", self.zdbase_fname), mmap_mode="r")
        self.fileidx_embidx_map = np.array(pickle.load(open(os.path.join(self.parent_dbase_dir, "metadata", self.fileidx_embidx_map_fname), "rb")))
        self.fileidx_cumsum_map = np.array(pickle.load(open(os.path.join(self.parent_dbase_dir, "metadata", self.fileidx_cumsum_map_fname), "rb")))
        self.refdbase_fnames = np.loadtxt(os.path.join(self.parent_dbase_dir, "metadata", self.reference_files_fname), dtype=str)
        self.ref_files_count_per_lang = pickle.load(open(os.path.join(self.parent_dbase_dir, "metadata", self.ref_files_count_per_lang_fname), "rb"))
        self.lang_index_ranges = pickle.load(open(os.path.join(self.parent_dbase_dir, "metadata", self.lang_startendidx_fname), "rb"))

        index_fnames = glob.glob(os.path.join(self.parent_dbase_dir, "index", "*.index"))
        self.index = {}
        for fname in index_fnames:
            lang = os.path.basename(fname).split("_tfidf")[0]
            self.index[lang] = faiss.read_index(fname)

        idf_count_fnames = glob.glob(os.path.join(self.parent_dbase_dir, "metadata", "idf_counts", "*.pkl"))
        self.idf_counts = {}
        for fname in idf_count_fnames:
            lang = os.path.basename(fname).split("_idf_count")[0]
            self.idf_counts[lang] = pickle.load(open(fname, "rb"))
        
        return 

    def tfidf_repr(self, query_tokens, lang="english", normalize=False):
        # use it for search purposes when indexing is done and idf_counts have been loaded to the memory
        tf_idf_vec = np.zeros(self.codebook_size, dtype=np.float16)
        u, c = np.unique(query_tokens, return_counts=True)
        c_sum = c.sum()
        dbase_size = self.get_dbase_size(lang)
        for i, token in enumerate(u):
            tf_idf_vec[token] = (c[i]/c_sum) * np.log(dbase_size/self.idf_counts[lang][token])
        
        if normalize:
            tf_idf_vec = tf_idf_vec/np.linalg.norm(tf_idf_vec) # ?? not sure to 
        
        return tf_idf_vec[None, :]

    def get_stats(self, I, groundtruth_fnames, verbose=False, search_lang="all_langs", beta=1):

        groundtruth_fnames = set(groundtruth_fnames)
        retrieved_fnames = set(self.refdbase_fnames[self.fileidx_embidx_map[I]])         
        recall = len(groundtruth_fnames.intersection(retrieved_fnames))/len(groundtruth_fnames)
        
        pmiss = (len(groundtruth_fnames) - len(groundtruth_fnames.intersection(retrieved_fnames)))/len(groundtruth_fnames)
        pfa = (len(retrieved_fnames) - len(groundtruth_fnames.intersection(retrieved_fnames)))/self.ref_files_count_per_lang[search_lang] 
        twv = 1-(pmiss + beta*pfa)

        if verbose:
            print(f"pool cand size: {len(retrieved_fnames)}")
            print(f"recall: {recall}")
            print(f"TWV: {twv}")
        return len(retrieved_fnames), recall, twv

    def get_metadata(self, retrieved_indices, hop=0.1):
        retrieved_fnames = self.refdbase_fnames[self.fileidx_embidx_map[retrieved_indices]]
        retrieved_timeoffset = hop*(retrieved_indices - self.fileidx_cumsum_map[self.fileidx_embidx_map[retrieved_indices]])
        retrieved_metadata = tuple(zip(retrieved_fnames.tolist(), retrieved_timeoffset.tolist()))
        return retrieved_metadata
    
    def search(self,
                query: List[np.ndarray],  
                search_lang: Optional[List[str]] = "all_langs",
                queries_fname: Optional[List[str]] = None, 
                index_probe: Optional[int]=20, 
                tfidf_topk: Optional[int]=10000,
                jaccard_thresh=0.2,
                edit_dist_thresh=0.4,
                dtw_quantz_thresh=0.2,
                dtw_z_thresh=0.15,
                codebook: Optional[torch.Tensor]=None,
                levels: Optional[int]=3,
                verbose: Optional[bool]=False,
                get_stats: Optional[bool]=False):

        
        tf_idf_queryvec, query_tokens, query_z = query 
        search_stages = np.zeros(5)
        search_stages[:levels] = 1
        search_stages = search_stages.astype(bool)

        assert search_lang in self.index, f"index not available for {search_lang}"
        index = self.index[search_lang]

        stats_log = []
        if search_stages[0]:
            # 1st pool of candidates using TF-IDF
            index.nprobe = index_probe       
            _, I = index.search(tf_idf_queryvec, tfidf_topk)     # search
            I = I[0]
            if search_lang != "all_langs":
                emb_idx_offset = self.lang_index_ranges[search_lang][0]
                I = I + emb_idx_offset

            if get_stats:
                candpoolsize, recall, twv = self.get_stats(I, queries_fname, verbose, search_lang)
                stats_log.extend([candpoolsize, recall, twv])
            if verbose:
                print("######## 1st stage filtering done ##########")

        if search_stages[1]:
            confidence = []
            # 2nd pool of candidates using jaccard similarity
            tokens_cands = self.tokens_dbase[I]
            filter_out_cands_mask = np.ones(len(I))
            for i, cand in enumerate(tokens_cands):
                sim = self.jaccard_similarity(cand, query_tokens)
                if sim < jaccard_thresh:
                    filter_out_cands_mask[i] = 0
                else:
                    confidence.append(sim)
            I = I[filter_out_cands_mask.astype(bool)]
            tokens_cands = tokens_cands[filter_out_cands_mask.astype(bool)]
            
            if get_stats:
                candpoolsize, recall, twv = self.get_stats(I, queries_fname, verbose, search_lang)
                stats_log.extend([candpoolsize, recall, twv])
            if verbose:
                print("######## 2nd stage filtering done ##########")

        if search_stages[2]:
            confidence = []
            # 3rd pool of candidates using edit distance of tokens sequence
            filter_out_cands_mask = np.ones(len(I))
            for i, token_cand in enumerate(tokens_cands):
                sim = ratio(torch.unique_consecutive(query_tokens), torch.unique_consecutive(torch.from_numpy(token_cand)))
                if sim < edit_dist_thresh:
                    filter_out_cands_mask[i] = 0
                else:
                    confidence.append(sim)
            I = I[filter_out_cands_mask.astype(bool)]
            tokens_cands = tokens_cands[filter_out_cands_mask.astype(bool)]
            
            if get_stats:
                candpoolsize, recall, twv = self.get_stats(I, queries_fname, verbose, search_lang)
                stats_log.extend([candpoolsize, recall, twv])
            if verbose:
                print("######## 3rd stage filtering done ##########")

        # if search_stages[3]:
        #     confidence = []
        #     # 4th pool of candidates using dtw on quantized represenations (deduplicated frames with same token index)
        #     filter_out_cands_mask = np.ones(len(I))
        #     assert codebook is not None, "codebook is None"

        #     query_quant_z = codebook[torch.unique_consecutive(query_tokens)]
        #     for i, token_cand in enumerate(tokens_cands):
        #         ref_quant_z = codebook[torch.unique_consecutive(token_cand)]
        #         _, dtw_cost = dtw_path_from_metric(query_quant_z.numpy(), ref_quant_z.numpy(), metric="cosine")
        #         dtw_cost = dtw_cost/(query_quant_z.shape[0]+ref_quant_z.shape[0])
        #         if dtw_cost > dtw_quantz_thresh:
        #             filter_out_cands_mask[i] = 0
        #         else:
        #             confidence.append(-dtw_cost)
        #     I = I[filter_out_cands_mask.astype(bool)]
        #     tokens_cands = tokens_cands[filter_out_cands_mask.astype(bool)]
            
        #     if get_stats:
        #         candpoolsize, recall, twv = self.get_stats(I, queries_fname, verbose, search_lang)
        #         stats_log.extend([candpoolsize, recall, twv])
        #     if verbose:
        #         print("######## 4th stage filtering done ##########")

        # if search_stages[4]:
        #     confidence = []
        #     filter_out_cands_mask = np.ones(len(I))
        #     z_cands = self.z_dbase[I]
        #     for i, ref_z in enumerate(z_cands):
        #         _, dtw_cost = dtw_path_from_metric(query_z.numpy(), ref_z, metric="cosine")
        #         dtw_cost = dtw_cost/(query_z.shape[0]+ref_z.shape[0])
        #         if dtw_cost > dtw_z_thresh:
        #             filter_out_cands_mask[i] = 0
        #         else:
        #             confidence.append(-dtw_cost)
        #     tokens_cands = tokens_cands[filter_out_cands_mask.astype(bool)]
        #     I = I[filter_out_cands_mask.astype(bool)]
            
        #     if get_stats:
        #         candpoolsize, recall, twv = self.get_stats(I, queries_fname, verbose, search_lang)
        #         stats_log.extend([candpoolsize, recall, twv])
        #     if verbose:
        #         print("######## 5th stage filtering done ##########")
            
        # sort matches based on confidence score and return its corresponding metadata
        sorted_idx = np.argsort(confidence)[::-1] 
        retrieved_indices = np.array(I)[sorted_idx]
        confidence = np.array(confidence)[sorted_idx]
        retrieved_metadata = self.get_metadata(retrieved_indices)
        
        retrieved_matches = {}
        for idx, item in enumerate(retrieved_metadata):
            fname, timeoffset = item[0], item[1]
            retrieved_matches[fname] = [timeoffset, confidence[idx]]

        if get_stats:
            return stats_log, retrieved_matches
        else:
            return retrieved_matches
        


    







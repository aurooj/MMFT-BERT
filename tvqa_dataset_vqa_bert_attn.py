import json

__author__ = "Jie Lei"
import io
import os
import h5py
import random
import numpy as np
from easydict import EasyDict as edict
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from utils import load_pickle, save_pickle, load_json, files_exist, get_show_name, get_random_chunk_from_ts, get_proper_nouns
from preprocessing import flip_question, get_qmask, get_verbs_and_nouns
from transformers import BertTokenizer
from preprocessing import get_split_index


class TVQADataset(Dataset):
    def __init__(self, opt, mode="train"):
        self.raw_train = load_json(opt.train_path)
        #self.raw_train = self.raw_train[:250]
        self.raw_test = load_json(opt.test_path)
        #self.raw_test = self.raw_test[:250]
        self.raw_valid = load_json(opt.valid_path)
        #self.raw_valid = self.raw_valid[:250]
        if isinstance(self.raw_train, dict):
            self.raw_train_full = self.raw_train
            self.raw_train = self.raw_train[opt.question_group]
        if isinstance(self.raw_test, dict):
            self.raw_test_full = self.raw_test
            self.raw_test = self.raw_test[opt.question_group]
        if isinstance(self.raw_valid, dict):
            self.raw_valid_full = self.raw_valid
            self.raw_valid = self.raw_valid[opt.question_group]

        self.vcpt_dict = load_pickle(opt.vcpt_path)
        self.vscene_dict = load_pickle(opt.vscene_path)
        self.bow = load_json(opt.bow_path)
        self.is_flip = opt.is_flip
        self.add_caption = opt.add_caption
        # attr_obj_pairs = self.get_obj_attr_pairs(self.vcpt_dict)

        self.vfeat_load = opt.vid_feat_flag
        self.vfeat = opt.vfeat
        self.vfeat_type = opt.vfeat_type
        self.word_drop = opt.word_drop
        if self.vfeat_load and self.vfeat not in ['i3d', 'c3d']:
            self.vid_h5 = h5py.File(opt.vid_feat_path, "r")
        if self.vfeat_load and self.vfeat in ['i3d', 'c3d']:
            self.vid_feat_path = opt.vid_feat_path
            with open('data/ts_2_chunks.json', 'rb') as f:
                self.ts2chunks = json.load(f)
                self.ts2chunks_dic = self.ts2chunks['ts_2_chunks_dic']
                # list of corrupt data points after removing tstamps
                bad_data = ["_".join(k.split("_")[:-2]) for k, v in self.ts2chunks_dic.items() if len(v) == 0]
                print("bad data points:{}".format(bad_data))

                # filter bad data from data:
                print("removing bad datapoints from\n"
                      " train {}, test {}, val sets {}..".format(len(self.raw_train), len(self.raw_test),
                                                                 len(self.raw_valid)))
                self.raw_train = filter(lambda x: x['vid_name'] not in bad_data, self.raw_train)
                self.raw_valid = filter(lambda x: x['vid_name'] not in bad_data, self.raw_valid)
                self.raw_test = filter(lambda x: x['vid_name'] not in bad_data, self.raw_test)
                print("After cleaning, data sizes are:\n"
                      " train {}, test {}, val sets {}..".format(len(self.raw_train),
                                                                 len(self.raw_test),
                                                                 len(self.raw_valid)))

        self.glove_embedding_path = opt.glove_path
        self.fasttext_embedding_path = opt.fasttext_path
        self.normalize_v = opt.normalize_v
        self.with_ts = opt.with_ts
        print("with_ts:{}".format(self.with_ts))
        self.mode = mode
        self.reverse = not opt.no_reverse if hasattr(opt, "no_reverse" ) else False
        print("reverse:{}".format(self.reverse))
        self.cur_data_dict = self.get_cur_dict()
        self.captions_folder_path = opt.captions_path

        self.pooled_feature_train = h5py.File('./data/train_res101_pooled_features.h5', "r")
        self.pooled_feature_train = self.pooled_feature_train['data']
        self.vid2idx_train = load_pickle('./data/train_res101_vid2idx.pkl')

        self.pooled_feature_valid = h5py.File('./data/valid_res101_pooled_features.h5', "r")
        self.pooled_feature_valid = self.pooled_feature_valid['data']
        self.vid2idx_valid = load_pickle('./data/valid_res101_vid2idx.pkl')

        self.pooled_feature, self.vid2idx = self.get_cur_pooled_feat()
        self.add_verbs_nouns = opt.add_verbs_nouns if hasattr(opt, 'add_verbs_nouns') else 1

        # self.num_blocks = opt.num_blocks

        # set word embedding / vocabulary
        self.word2idx_path = opt.word2idx_path
        self.idx2word_path = opt.idx2word_path
        self.vocab_embedding_path = opt.vocab_embedding_path
        self.vocab_embedding_path2 = opt.vocab_embedding_path2
        self.bert_path = opt.bert_path
        self.embedding_dim = opt.embedding_size
        self.word2idx = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
        self.idx2word = {0: "<pad>", 1: "<unk>", 2: "<eos>"}
        self.offset = len(self.word2idx)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer_opt = "bert"
        self.max_cap_l = opt.max_cap_l
        self.max_seq_len = opt.max_seq_len

        # set entry keys
        if self.with_ts:
            self.text_keys = ["q", "a0", "a1", "a2", "a3", "a4", "located_sub_text"]
        else:
            self.text_keys = ["q", "a0", "a1", "a2", "a3", "a4", "sub_text"]
        self.vcpt_key = "vcpt"
        self.label_key = "answer_idx"
        self.qid_key = "qid"
        self.vid_name_key = "vid_name"
        self.located_frm_key = "located_frame"
        for k in self.text_keys + [self.vcpt_key, self.qid_key, self.vid_name_key]:
            if k == "vcpt":
                continue
            assert k in self.raw_valid[0].keys()

        # build/load vocabulary
        if not files_exist(
                [self.word2idx_path, self.idx2word_path, self.vocab_embedding_path, self.vocab_embedding_path]):
            print("\nNo cache founded.")
            self.build_word_vocabulary(word_count_threshold=opt.word_count_threshold)
        else:
            print("\nLoading cache ...")
            self.word2idx = load_pickle(self.word2idx_path)
            self.idx2word = load_pickle(self.idx2word_path)

            # if opt.no_glove:
            #     if not files_exist([self.bert_path]):
            #         print("No cache found for bert vocab.")
            #         self.bert_embedding = self.load_bert(self.word2idx)
            #     else:
            #         self.bert_embedding = load_pickle(self.bert_path)
            # else:
            #     self.vocab_embedding = load_pickle(self.vocab_embedding_path)

    def set_mode(self, mode):
        self.mode = mode
        self.cur_data_dict = self.get_cur_dict()
        self.pooled_feature, self.vid2idx = self.get_cur_pooled_feat()
        self.captions_path = os.path.join(self.captions_folder_path, self.mode)

    def get_obj_attr_pairs(self, vcpts):
        attributes = []
        attr_obj_pairs = []
        for clip in tqdm(self.vcpt_dict.values()):
            for item in clip:
                attr_obj_pairs.extend(filter(lambda x: len(x.split()) > 1, item))
        return attr_obj_pairs

    def get_cur_dict(self):
        if self.mode == 'train':
            return self.raw_train
        elif self.mode == 'valid':
            return self.raw_valid
        elif self.mode == 'test':
            return self.raw_test

    def get_cur_pooled_feat(self):
        if self.mode == 'train':
            return self.pooled_feature_train, self.vid2idx_train
        elif self.mode == 'valid':
            return self.pooled_feature_valid, self.vid2idx_valid
        elif self.mode == 'test':
            return self.pooled_feature_valid, self.vid2idx_valid

    def __len__(self):
        return len(self.cur_data_dict)

    def __getitem__(self, index):
        items = edict()

        qid = str(self.cur_data_dict[index]['qid'])
        cur_start, cur_end = self.cur_data_dict[index][self.located_frm_key]
        cur_vid_name = self.cur_data_dict[index][self.vid_name_key]

        # add text keys
        for k in self.text_keys:
            if self.word_drop:
                items.append(self.word_dropout(self.numericalize(self.cur_data_dict[index][k])))
            else:
                if k is 'q':
                    q = self.numericalize(self.cur_data_dict[index][k])

                    if self.mode == 'train' and self.is_flip:
                        # if np.random.rand() > 0.5:
                        if random.getrandbits(1):

                            q_fl, qmask1, qmask2 = flip_question(q, len(q), tokenizer=self.tokenizer_opt)
                            if qmask1 is None and qmask2 is None:
                                qmask1, qmask2, split_idx = get_qmask(q, len(q), tokenizer=self.tokenizer_opt)
                            items['q'] = q_fl

                        else:
                            qmask1, qmask2, split_idx = get_qmask(q, len(q), tokenizer=self.tokenizer_opt)
                            items['q'] = q
                    else:
                        qmask1, qmask2, split_idx = get_qmask(q, len(q), tokenizer=self.tokenizer_opt)
                        items['q'] = q
                    items['qmask1'] = qmask1
                    items['qmask2'] = qmask2

                else:
                    if k in ['located_sub_text', 'sub_text']: #for now using same key  for located and full sub text
                        located_sub = self.numericalize_sub(self.cur_data_dict[index][k])
                        located_sub_mask = [1] * len(located_sub)
                        items['located_sub_text'] = located_sub
                        items['located_sub_mask'] = located_sub_mask
                    else:
                        question = self.cur_data_dict[index]['q']
                        split_idx = get_split_index(self.tokenizer.encode(question, add_special_tokens=False),
                                                    tokenizer=self.tokenizer_opt)
                        q_words = question.split(" ")
                        if split_idx != -1:
                            q1 = " ".join(q_words[:split_idx])
                            q2 = " ".join(q_words[split_idx:])
                        else:
                            q1 = " ".join(q_words[:len(q_words) // 2])
                            q2 = " ".join(q_words[len(q_words) // 2:])
                        if self.add_verbs_nouns:
                            verbs, nouns = get_verbs_and_nouns(question)
                            input = q1 + " [SEP] " + q2 + " [SEP] " + " ".join(verbs) + " [SEP] " + " ".join(
                            nouns) + " [SEP] " + self.cur_data_dict[index][k]
                        else:
                            input = question + " [SEP] " + self.cur_data_dict[index][k]
                        if self.with_ts:
                            cur_vis_sen = self.vcpt_dict[cur_vid_name][cur_start:cur_end + 1]
                            if len(cur_vis_sen) == 0:
                                cur_vis_sen = self.vcpt_dict[cur_vid_name]
                        else:
                            cur_vis_sen = self.vcpt_dict[cur_vid_name]
                        cur_vis_sen = ", ".join(cur_vis_sen)
                        cur_vis_sen = self.get_unique_words(cur_vis_sen)
                        _ = random.shuffle(cur_vis_sen)
                        if self.add_caption:    #get captions

                            captions = load_json(os.path.join(self.captions_path, cur_vid_name+"_"+ qid +".json"))[2]
                            captions = " ".join(" , ".join(captions).split(" ")[:self.max_cap_l])
                            vqa = " ".join(cur_vis_sen) + " [$] caption:" + captions + " [$] " + input
                        else:
                            subtitles =  " ".join(self.line_to_words(self.cur_data_dict[index]['located_sub_text']))

                            snouns = get_proper_nouns(subtitles)
                            vqa = " ".join(cur_vis_sen) + " [$] " + input
                            if self.with_ts:
                                subtitles =  " ".join(self.line_to_words(self.cur_data_dict[index]['located_sub_text']))
                            else:
                                subtitles =  " ".join(self.line_to_words(self.cur_data_dict[index]['sub_text']))

                            sqa = subtitles + " [$] " + input

                        vqa = self.trunc_if_gt_max_seq_len(vqa, self.reverse, self.max_seq_len)
                        sqa = self.trunc_if_gt_max_seq_len(sqa, self.reverse, self.max_seq_len)
                        #print(sqa)
                        vqa_tokens = self.numericalize(vqa)
                        vqa_mask = [1] * len(vqa_tokens)
                        sqa_tokens = self.numericalize(sqa)
                        sqa_mask = [1] * len(sqa_tokens)

                        items['vq' + k] = vqa_tokens
                        items['vq'+ k +'_mask'] = vqa_mask
                        items['sq' + k] = sqa_tokens
                        items['sq'+ k +'_mask'] = sqa_mask


                        input_tokens = self.numericalize(input)
                        input_mask = [1] * len(input_tokens)
                        items['q'+k] = input_tokens
                        items['q'+k+'_mask'] = input_mask

        # add vcpt
        if self.with_ts:
            cur_vis_sen = self.vcpt_dict[cur_vid_name][cur_start:cur_end + 1]
            cur_vscene_sen = self.vscene_dict[cur_vid_name][cur_start:cur_end + 1]
            ts = list(range(cur_start, cur_end + 1))
            normalized_ts = self.normalize_tstamps(ts)
            vcpt, vcpt_ts = self.numericalize_vcpts_with_ts(cur_vis_sen, normalized_ts)
            items['vcpt'] = vcpt
            items['vcpt_ts'] = vcpt_ts

            vscene, vscene_ts = self.numericalize_vcpts_with_ts(cur_vscene_sen, normalized_ts)
            items['vscene'] = vscene
            items['vscene_ts'] = vscene_ts

        else:
            cur_vis_sen = self.vcpt_dict[cur_vid_name]
            cur_vscene_sen = self.vscene_dict[cur_vid_name]
            cur_vis_sen = " , ".join(cur_vis_sen)
            cur_vscene_sen = " , ".join(cur_vscene_sen)

            items['vcpt'] = self.numericalize_vcpt(cur_vis_sen)
            items['vcpt_ts'] = [1, 1] # dummy value

            items['vscene'] = self.numericalize_vcpt(cur_vscene_sen)
            items['vscene_ts'] = [1, 1] # dummy value


        # add other keys
        if self.mode == 'test':
            # items[self.label_key] = 666 # this value will not be used
            items[self.label_key] = int(self.cur_data_dict[index][self.label_key])
        else:
            items[self.label_key] = int(self.cur_data_dict[index][self.label_key])

        for k in [self.qid_key]:
            items[self.qid_key] = self.cur_data_dict[index][k]

        items['vid_name'] = cur_vid_name

        # add visual feature
        if self.vfeat_load and self.vfeat not in ['i3d', 'c3d']:
            if self.with_ts:
                cur_vid_feat = torch.from_numpy(self.vid_h5[cur_vid_name][cur_start:cur_end])
            else:  # handled by vid_path
                cur_vid_feat = torch.from_numpy(self.vid_h5[cur_vid_name][:480])
            if self.normalize_v:
                cur_vid_feat = nn.functional.normalize(cur_vid_feat, p=2, dim=1)

        elif self.vfeat_load and self.vfeat == 'c3d':
            show = get_show_name(cur_vid_name)
            path = os.path.join(self.vid_feat_path, '{}_frames'.format(show), cur_vid_name,
                                '{}_{}_{}_{}.npy'.format(cur_vid_name, cur_vid_name, cur_start, cur_end))

            cur_vid_feat = torch.from_numpy(np.load(os.path.join(path)))

        elif self.vfeat_load and self.vfeat == 'i3d':
            show = get_show_name(cur_vid_name)
            path = os.path.join(self.vid_feat_path, '{}_frames'.format(show), '{}.npy'.format(cur_vid_name))
            # if self.with_ts:
            #     cur_vid_feat = torch.from_numpy(np.load(path)[cur_start:cur_end])
            # else:  # handled by vid_path
            cur_vid_feat = torch.from_numpy(np.load(path))
            # randomly sample consecutive num_blocks from video features
            chunks_list_key = '{}_{}_{}'.format(cur_vid_name, cur_start, cur_end)
            chunks_list = self.ts2chunks_dic[chunks_list_key]
           
            if self.with_ts:
            # else:  # handled by vid_path
                cur_vid_feat = torch.from_numpy(np.load(path))
            # randomly sample consecutive num_blocks from video features
            chunks_list_key = '{}_{}_{}'.format(cur_vid_name, cur_start, cur_end)
            chunks_list = self.ts2chunks_dic[chunks_list_key]
            if self.with_ts:
                assert len(chunks_list) != 0, "chunks_list is zero for {} with ts:{},{}".format(cur_vid_name, cur_start,
                                                                                                cur_end)

                if len(chunks_list) > 15 and self.vfeat_type == "conv":
                    cur_vid_feat = cur_vid_feat[chunks_list[:12]]
                else:
                    cur_vid_feat = cur_vid_feat[chunks_list]  # localized features

                    # chunk, cur_vid_feat = self.get_features_for_fixed_blks(chunks_list, cur_vid_feat)
            else:
                ts = [chunks_list[0], chunks_list[-1]]  # ts in terms of clips
                # num_blocks are larger here: 14 avg length
                chunk = get_random_chunk_from_ts(ts, self.num_blocks, len(cur_vid_feat))
                # assert len(chunk) == self.num_blocks, "length of chunk: {} should equals num_blocks{}".format(chunk, self.num_blocks)
                cur_vid_feat = cur_vid_feat[chunk[0]:chunk[-1] + 1]
            if self.normalize_v:
                cur_vid_feat = nn.functional.normalize(cur_vid_feat, p=2, dim=1)
        else:
            try:
                pool_idx = self.vid2idx[cur_vid_name + "_" + qid]
                cur_vid_feat = torch.from_numpy(self.pooled_feature[pool_idx]).permute(1,2,0)
            except:
                cur_vid_feat = torch.zeros(14,14,1024) 
        items['vid_feat'] = cur_vid_feat

        return items

    def word_dropout(self, input, dropout=0.1):
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input)
        if isinstance(input, list):
            input = torch.as_tensor(input)
        # probabilities
        probs = torch.empty(input.size(0)).uniform_(0, 1)

        # app word dropout
        input = torch.where(probs > dropout, input, torch.zeros(input.size(0), dtype=torch.long))
        return input

    def get_features_for_fixed_blks(self, chunks_list, cur_vid_feat):
        if len(cur_vid_feat) < self.num_blocks:
            sz = cur_vid_feat.size()
            test = torch.zeros((self.num_blocks, sz[1], sz[2], sz[3], sz[4]))
            test[:len(cur_vid_feat)] = cur_vid_feat
            cur_vid_feat = test
        if len(chunks_list) != 0:
            ts = [chunks_list[0], chunks_list[-1]]
        else:
            ts = [0, len(cur_vid_feat)]
        chunk = get_random_chunk_from_ts(ts, self.num_blocks, len(cur_vid_feat))
        return chunk, cur_vid_feat

    @classmethod
    def line_to_words(cls, line, eos=True, downcase=True):
        eos_word = "<eos>"
        words = line.lower().split() if downcase else line.split()
        # !!!! remove comma here, since they are too many of them
        words = [w for w in words if w != ","]
        words = words + [eos_word] if eos else words
        return words

    def numericalize(self, sentence, eos=True):
        """convert words to indices"""
        # sentence_indices = [self.word2idx[w] if w in self.word2idx else self.word2idx["<unk>"]
        #                     for w in self.line_to_words(sentence, eos=eos)]  # 1 is <unk>, unknown
        sentence_indices = self.tokenizer.encode(sentence, add_special_tokens=True)
        return sentence_indices

    def numericalize_sub(self, sentence, eos=True):
        """convert words to indices"""
        sentence_indices = [self.word2idx[w] if w in self.word2idx else self.word2idx["<unk>"]
                            for w in self.line_to_words(sentence, eos=eos)]  # 1 is <unk>, unknown
#        sentence = " ".join(self.line_to_words(sentence, eos=eos))
#        sentence_indices = self.tokenizer.encode(sentence, add_special_tokens=True)
        return sentence_indices

    def numericalize_vcpt(self, vcpt_sentence):
        """convert words to indices, additionally removes duplicated attr-object pairs"""

        words = self.get_unique_words(vcpt_sentence)
        sentence_indices = [self.word2idx[w] if w in self.word2idx else self.word2idx["<unk>"]
                            for w in words]
        # sentence_indices = self.tokenizer.encode(vcpt_sentence, add_special_tokens=True)
        return sentence_indices

    def get_unique_words(self, attr_obj_pairs):
        attr_obj_pairs = attr_obj_pairs.lower().split(",")  # comma is also removed
        unique_pairs = []
        pair_freq = {}
        for pair in attr_obj_pairs:
            pair = pair.strip()
            if pair not in unique_pairs:
                pair_freq[pair] = 1
                unique_pairs.append(pair)
            else:
                pair_freq[pair] += 1
        words = []
        for pair in unique_pairs:
            words.extend(pair.split())
        words.append("<eos>")

        # word_freq = {w:v for w in k.split() for k,v in pair_freq.items()}

        return words

    def normalize_tstamps(self, timestamps):
        # return dictionary with original ts(key), normalized ts (value) pairs
        min_ts = min(timestamps)
        max_ts = max(timestamps)
        normalized_ts = {k: (k - min_ts) / (max_ts - min_ts) for k in timestamps}
        return normalized_ts

    def numericalize_vcpts_with_ts(self, vcpt_sentences, normalized_ts):
        unique_pairs = []
        unique_pairs_with_ts = []
        ts = []

        keys = list(normalized_ts.keys())
        cur_vis = [c.lower().split(",") for c in vcpt_sentences]
        # cur_vis = cur_vis[keys[0]:keys[-1]+1] #to take care of out of index error
        for k, pair_list in enumerate(cur_vis):
            for pair in pair_list:
                pair = pair.strip()
                if pair not in unique_pairs:
                    unique_pairs.append(pair)
                    unique_pairs_with_ts.append([pair, normalized_ts[keys[k]]])

        words = []
        for pair in unique_pairs_with_ts:
            lst = pair[0].split()
            words.extend(lst)
            ts.extend([pair[1]] * len(lst))
        words.append("<eos>")
        ts.append(-1.0)  # dummy value for <eos>

        sentence_indices = [self.word2idx[w] if w in self.word2idx else self.word2idx["<unk>"]
                            for w in words]
        # sentence_indices = self.tokenizer.encode(vcpt_sentences, add_special_tokens=True)
        return sentence_indices, ts

    def load_fasttext(cls, fname):
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = map(float, tokens[1:])
        return data

    @classmethod
    def load_glove(cls, filename):
        """ Load glove embeddings into a python dict
        returns { word (str) : vector_embedding (torch.FloatTensor) }"""
        glove = {}
        with open(filename) as f:
            for line in f.readlines():
                values = line.strip("\n").split(" ")  # space separator
                word = values[0]
                vector = np.asarray([float(e) for e in values[1:]])
                glove[word] = vector
        return glove

    def load_bert(self, word2idx):
        bert_matrix = np.zeros([len(self.idx2word), 1024])

        for i, w in tqdm(enumerate(word2idx.keys())):
            bert_matrix[i, :] = self.sentence_encoder.encode([w])[0]
        print("Bert embedding size is:", bert_matrix.shape)

        print("Saving cache files ...\n")

        save_pickle(bert_matrix, self.bert_path)
        # save_pickle(fast_matrix, self.vocab_embedding_path2)
        print("Building bert vocabulary done.\n")
        return bert_matrix

    def build_word_vocabulary(self, word_count_threshold=0):
        """borrowed this implementation from @karpathy's neuraltalk."""
        print("Building word vocabulary starts.\n")
        all_sentences = []
        for k in self.text_keys:
            all_sentences.extend([ele[k] for ele in self.raw_train])

        word_counts = {}
        for sentence in all_sentences:
            for w in self.line_to_words(sentence, eos=False, downcase=True):
                word_counts[w] = word_counts.get(w, 0) + 1

        vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold and w not in self.word2idx.keys()]
        print("Vocabulary Size %d (<pad> <unk> <eos> excluded) using word_count_threshold %d.\n" %
              (len(vocab), word_count_threshold))

        # build index and vocabularies
        for idx, w in enumerate(vocab):
            self.word2idx[w] = idx + self.offset
            self.idx2word[idx + self.offset] = w
        print("word2idx size: %d, idx2word size: %d.\n" % (len(self.word2idx), len(self.idx2word)))


        # Make glove embedding.
        print("Loading glove embedding at path : %s. \n" % self.glove_embedding_path)
        glove_full = self.load_glove(self.glove_embedding_path)
        print("Glove Loaded, building word2idx, idx2word mapping. This may take a while.\n")
        glove_matrix = np.zeros([len(self.idx2word), self.embedding_dim])
        glove_keys = glove_full.keys()
        for i in tqdm(range(len(self.idx2word))):
            w = self.idx2word[i]
            w_embed = glove_full[w] if w in glove_keys else np.random.randn(self.embedding_dim) * 0.4
            glove_matrix[i, :] = w_embed
        self.vocab_embedding = glove_matrix
        print("Vocab embedding size is :", glove_matrix.shape)

        print("Saving cache files ...\n")
        save_pickle(self.word2idx, self.word2idx_path)
        save_pickle(self.idx2word, self.idx2word_path)
        save_pickle(glove_matrix, self.vocab_embedding_path)
        # save_pickle(fast_matrix, self.vocab_embedding_path2)
        print("Building  vocabulary done.\n")

    def trunc_if_gt_max_seq_len(self, vqa, reverse, max_seq_len=300):
        adjust_len_idx = lambda x: x*-1 if (x < 0) else x
        adjust_v_len = lambda v, idx: v.split(" ")[:-idx]

        full_len = len(vqa.split(" "))
        if full_len > max_seq_len:
            vqa_split = vqa.split(" [$] ")
            v = vqa_split[0]

            if self.add_caption: #captions included
                len_adjust_idx = full_len - max_seq_len + 2 + len(vqa_split[1].split(" ")) + len(vqa_split[2].split(" "))
                len_adjust_idx = adjust_len_idx(len_adjust_idx)
                v = adjust_v_len(v, len_adjust_idx)
                vqa = " ".join(v) + " [SEP] " + vqa_split[1] + " . " + vqa_split[2]

            else: #no captions
                len_adjust_idx = full_len - max_seq_len + 2 + len(vqa_split[1].split(" "))
                len_adjust_idx = adjust_len_idx(len_adjust_idx)
                v = adjust_v_len(v, len_adjust_idx)
                if reverse:
                    vqa = vqa_split[1] + " . " + " ".join(v)  #put qa pair first
                else:
                    vqa = " ".join(v) + " . " + vqa_split[1]

            if len(vqa.split(" ")) >= max_seq_len:
                print(len(vqa.split(" ")))
                vqa = " ".join(vqa.split(" ")[:max_seq_len])
        else:
            vqa_split = vqa.split(" [$] ")
            if reverse:
                vqa = vqa_split[1] + " . " + vqa_split[0]
            else:
                vqa = " . ".join(vqa_split)
        return vqa

def pad_collate(data):

    def pad_sequences(sequences, key=None):
        if key is not None and key in ["vcpt_ts", "vscene_ts"]:
            sequences = [torch.FloatTensor(s) for s in sequences]
            lengths = torch.LongTensor([len(seq) for seq in sequences])
            padded_seqs = torch.zeros(len(sequences), max(lengths))
        else:
            sequences = [torch.LongTensor(s) for s in sequences]
            lengths = torch.LongTensor([len(seq) for seq in sequences])
            padded_seqs = torch.zeros(len(sequences), max(lengths)).long()

        for idx, seq in enumerate(sequences):
            end = lengths[idx]
            padded_seqs[idx, :end] = seq[:end]
        return padded_seqs, lengths

    def pad_video_sequences(sequences):
        """sequences is a list of torch float tensors (created from numpy)"""
        lengths = torch.LongTensor([len(seq) for seq in sequences])
        v_dim = sequences[0].size(1)
        padded_seqs = torch.zeros(len(sequences), max(lengths), v_dim).float()
        length_masks = torch.zeros(len(sequences), max(lengths))
        for idx, seq in enumerate(sequences):
            end = lengths[idx]
            padded_seqs[idx, :end] = seq
            length_masks[idx, :end] = torch.ones(end.item())
        return padded_seqs, lengths, length_masks

    def pad_video_tensors(sequences):
        """sequences is a list of torch float multi-dimensional tensors (created from numpy)"""
        lengths = torch.LongTensor([len(seq) for seq in sequences])
        v_l, f, t, h, w = sequences[0].size()
        padded_seqs = torch.zeros(len(sequences), max(lengths), f, t, h, w).float()
        length_masks = torch.zeros(len(sequences), max(lengths))
        for idx, seq in enumerate(sequences):
            end = lengths[idx]
            padded_seqs[idx, :end] = seq
            length_masks[idx, :end] = torch.ones(end.item())
        return padded_seqs, lengths, length_masks

    # separate source and target sequences
    batch = {}#edict()

    batch['q'], batch['q_l'] = pad_sequences([d['q'] for d in data])
    batch['qmask1'], batch['qm1_l']  = pad_sequences([d['qmask1'] for d in data])
    batch['qmask2'], batch['qm2_l']  = pad_sequences([d['qmask2'] for d in data])

    batch['vqa0'], batch['vqa0_l']  = pad_sequences([d['vqa0'] for d in data])
    batch['vqa0_mask'], _  = pad_sequences([d['vqa0_mask'] for d in data])
    batch['sqa0'], batch['sqa0_l']  = pad_sequences([d['sqa0'] for d in data])
    batch['sqa0_mask'], _  = pad_sequences([d['sqa0_mask'] for d in data])
    batch['qa0'], batch['qa0_l']  = pad_sequences([d['qa0'] for d in data])
    batch['qa0_mask'], _  = pad_sequences([d['qa0_mask'] for d in data])

    batch['vqa1'], batch['vqa1_l']  = pad_sequences([d['vqa1'] for d in data])
    batch['vqa1_mask'], _  = pad_sequences([d['vqa1_mask'] for d in data])
    batch['sqa1'], batch['sqa1_l']  = pad_sequences([d['sqa1'] for d in data])
    batch['sqa1_mask'], _  = pad_sequences([d['sqa1_mask'] for d in data])
    batch['qa1'], batch['qa1_l']  = pad_sequences([d['qa1'] for d in data])
    batch['qa1_mask'], _  = pad_sequences([d['qa1_mask'] for d in data])

    batch['vqa2'], batch['vqa2_l']  = pad_sequences([d['vqa2'] for d in data])
    batch['vqa2_mask'], _  = pad_sequences([d['vqa2_mask'] for d in data])
    batch['sqa2'], batch['sqa2_l']  = pad_sequences([d['sqa2'] for d in data])
    batch['sqa2_mask'], _  = pad_sequences([d['sqa2_mask'] for d in data])
    batch['qa2'], batch['qa2_l']  = pad_sequences([d['qa2'] for d in data])
    batch['qa2_mask'], _  = pad_sequences([d['qa2_mask'] for d in data])

    batch['vqa3'], batch['vqa3_l']  = pad_sequences([d['vqa3'] for d in data])
    batch['vqa3_mask'], _  = pad_sequences([d['vqa3_mask'] for d in data])
    batch['sqa3'], batch['sqa3_l']  = pad_sequences([d['sqa3'] for d in data])
    batch['sqa3_mask'], _  = pad_sequences([d['sqa3_mask'] for d in data])
    batch['qa3'], batch['qa3_l']  = pad_sequences([d['qa3'] for d in data])
    batch['qa3_mask'], _  = pad_sequences([d['qa3_mask'] for d in data])

    batch['vqa4'], batch['vqa4_l']  = pad_sequences([d['vqa4'] for d in data])
    batch['vqa4_mask'], _  = pad_sequences([d['vqa4_mask'] for d in data])
    batch['sqa4'], batch['sqa4_l']  = pad_sequences([d['sqa4'] for d in data])
    batch['sqa4_mask'], _  = pad_sequences([d['sqa4_mask'] for d in data])
    batch['qa4'], batch['qa4_l']  = pad_sequences([d['qa4'] for d in data])
    batch['qa4_mask'], _  = pad_sequences([d['qa4_mask'] for d in data])

    batch['sub_text'], batch['sub_l']  = pad_sequences([d['located_sub_text'] for d in data])
    batch['sub_mask'], _  = pad_sequences([d['located_sub_mask'] for d in data])

    batch['vcpt'], batch['vcpt_l']  = pad_sequences([d['vcpt'] for d in data])
    batch['vcpt_ts'], _  = pad_sequences([d['vcpt_ts'] for d in data], key="vcpt_ts")
    batch['vscene'], batch['vscene_l']  = pad_sequences([d['vscene'] for d in data])
    batch['vscene_ts'], _  = pad_sequences([d['vscene_ts'] for d in data], key='vscene_ts')

    batch["answer_idx"] = torch.LongTensor([d['answer_idx'] for d in data])
    batch["pooled_spatiotemp"] = torch.stack([d['vid_feat'] for d in data], dim=0)
    batch["qid"] = [d['qid'] for d in data]
    batch["vid_name"] =  [d['vid_name'] for d in data]

    return batch


def preprocess_inputs(batch, max_len_dict, device="cuda:0"):
    """clip and move to target device"""

    label_key = "answer_idx"
    qid_key = "qid"
    ctx_keys = ['vcpt', 'vcpt_ts', 'vscene', 'vscene_ts']
    # vfeat_type = "conv"
    model_in_dict = {}#edict()

    model_in_dict["q"] = batch["q"].to(device)
    model_in_dict["q_l"] = batch["q_l"].to(device)
    model_in_dict["qmask1"], model_in_dict["qm1_l"] = batch['qmask1'].to(device), batch['qm1_l'].to(device)
    model_in_dict["qmask2"], model_in_dict["qm2_l"] = batch['qmask2'].to(device), batch['qm2_l'].to(device)

    for k in range(5):
        i = str(k)
        model_in_dict["vqa{}".format(i)] = batch["vqa{}".format(i)][:, :max_len_dict["max_seq_len"]].to(device)
        model_in_dict["vqa{}_l".format(i)] = batch['vqa{}_l'.format(i)].clamp(min=1, max=max_len_dict["max_seq_len"]).to(device)
        model_in_dict["vqa{}_mask".format(i)] = batch['vqa{}_mask'.format(i)][:, :max_len_dict["max_seq_len"]].to(device)

        model_in_dict["sqa{}".format(i)] = batch["sqa{}".format(i)][:, :max_len_dict["max_seq_len"]].to(device)
        model_in_dict["sqa{}_l".format(i)] = batch['sqa{}_l'.format(i)].clamp(min=1, max=max_len_dict["max_seq_len"]).to(device)
        model_in_dict["sqa{}_mask".format(i)] = batch['sqa{}_mask'.format(i)][:, :max_len_dict["max_seq_len"]].to(device)   

        model_in_dict["qa{}".format(i)] = batch['qa{}'.format(i)][:, :max_len_dict["max_seq_len"]].to(device)
        model_in_dict["qa{}_l".format(i)] = batch['qa{}_l'.format(i)].clamp(min=1, max=max_len_dict["max_seq_len"]).to(device)
        model_in_dict["qa{}_mask".format(i)] = batch['qa{}_mask'.format(i)][:, :max_len_dict["max_seq_len"]].to(device)

    model_in_dict["sub_text"] = batch["sub_text"][:, :max_len_dict["sub"]].to(device)
    model_in_dict["sub_l"] = batch["sub_l"].clamp(min=1, max=max_len_dict["sub"]).to(device)
    model_in_dict["sub_mask".format(i)] = batch['sub_mask'.format(i)][:, :max_len_dict["sub"]].to(device)

    for k in ctx_keys:
        max_l = min(batch[k].shape[1], max_len_dict[k])
        model_in_dict[k] = batch[k][:, :max_l].to(device)

    model_in_dict["vcpt_l"] = batch["vcpt_l"].clamp(min=1, max=max_len_dict["vcpt"]).to(device)
    model_in_dict["vscene_l"] = batch["vcpt_l"].clamp(min=1, max=max_len_dict["vcpt"]).to(device) #vscene has same len as vcpt
    model_in_dict["pooled_spatiotemp"] = batch["pooled_spatiotemp"].to(device)

    target_data = batch[label_key]
    target_data = target_data.to(device)

    qid_data = batch[qid_key]
#    if torch.cuda.is_available():
#        for k,v in model_in_dict.items():
#            model_in_dict[k].cuda(device=[0,1])
#        target_data.cuda(device=[0,1])
    return model_in_dict, target_data, qid_data


if __name__ == "__main__":
    # python tvqa_dataset.py --input_streams sub
    import sys
    from config import BaseOptions

    sys.argv[1:] = ["--input_streams", "sub"]
    opt = BaseOptions().parse()

    dset = TVQADataset(opt, mode="valid")
    data_loader = DataLoader(dset, num_workers=16, batch_size=10, shuffle=False, collate_fn=pad_collate)

    for batch_idx, batch in enumerate(data_loader):
        model_inputs, targets, qids = preprocess_inputs(batch, opt.max_sub_l, opt.max_vcpt_l, opt.max_vid_l)
        break

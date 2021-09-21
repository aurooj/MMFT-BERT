import json

__author__ = "Jie Lei"
import io
import os
import h5py
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from utils import load_pickle, save_pickle, load_json, files_exist, get_show_name, get_random_chunk_from_ts
from preprocessing import flip_question, get_qmask
from transformers import BertTokenizer


class TVQADataset(Dataset):
    def __init__(self, opt, mode="train"):
        self.raw_train = load_json(opt.train_path)
        # self.raw_train = self.raw_train[:250]
        self.raw_test = load_json(opt.test_path)
        self.raw_valid = load_json(opt.valid_path)
        # self.raw_valid = self.raw_valid[:250]
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
        # attr_obj_pairs = self.get_obj_attr_pairs(self.vcpt_dict)

        self.vfeat_load = opt.vid_feat_flag
        self.vfeat = opt.vfeat
        self.vfeat_type = opt.vfeat_type
        self.word_drop = opt.word_drop
        if self.vfeat_load and self.vfeat not in ['i3d', 'c3d']:
            self.vid_h5 = h5py.File(opt.vid_feat_path, "r")
        if self.vfeat_load and self.vfeat in ['i3d','c3d']:
            self.vid_feat_path = opt.vid_feat_path
            with open('data/ts_2_chunks.json', 'rb') as f:
                self.ts2chunks = json.load(f)
                self.ts2chunks_dic = self.ts2chunks['ts_2_chunks_dic']
                #list of corrupt data points after removing tstamps
                bad_data = ["_".join(k.split("_")[:-2]) for k, v in self.ts2chunks_dic.items() if len(v) == 0]
                print("bad data points:{}".format(bad_data))

                #filter bad data from data:
                print("removing bad datapoints from\n"
                      " train {}, test {}, val sets {}..".format(len(self.raw_train), len(self.raw_test), len(self.raw_valid)))
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
        self.cur_data_dict = self.get_cur_dict()
        self.num_blocks = opt.num_blocks

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

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', max_seq_len=128)

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
        if not files_exist([self.word2idx_path, self.idx2word_path, self.vocab_embedding_path, self.vocab_embedding_path]):
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

    def __len__(self):
        return len(self.cur_data_dict)

    def __getitem__(self, index):
        items = []
        # if self.with_ts: #changed by me to get tstamps anyway
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
                            # print(len(q))
                            # print(len(flip_question(q, len(q))[0]))
                            q_fl, qmask1, qmask2 = flip_question(q, len(q))
                            if qmask1 is None and qmask2 is None:
                                qmask1, qmask2 = get_qmask(q, len(q))
                            items.extend([q_fl, qmask1, qmask2])
                        else:
                            qmask1, qmask2, split_idx = get_qmask(q, len(q))
                            items.extend([q, qmask1, qmask2])
                    else:
                        qmask1, qmask2, split_idx = get_qmask(q, len(q))
                        _, q1_a0, q2_a0 = self.numericalize_qa(self.cur_data_dict[index]['q'], self.cur_data_dict[index]['a0'], split_idx)
                        _, q1_a1, q2_a1 = self.numericalize_qa(self.cur_data_dict[index]['q'], self.cur_data_dict[index]['a1'], split_idx)
                        _, q1_a2, q2_a2 = self.numericalize_qa(self.cur_data_dict[index]['q'], self.cur_data_dict[index]['a2'], split_idx)
                        _, q1_a3, q2_a3 = self.numericalize_qa(self.cur_data_dict[index]['q'], self.cur_data_dict[index]['a3'], split_idx)
                        _, q1_a4, q2_a4 = self.numericalize_qa(self.cur_data_dict[index]['q'], self.cur_data_dict[index]['a4'], split_idx)
                        items.extend([q1_a0, q2_a0, q1_a1, q2_a1, q1_a2, q2_a2, q1_a3, q2_a3, q1_a4, q2_a4])
                    # if len(q) != len(qmask1):
                    #     print(q, qmask1)
                else:
                    if k is 'located_sub_text':
                        items.append(self.numericalize_sub(self.cur_data_dict[index][k]))
                    else:
                        input = self.cur_data_dict[index]['q'] + " [SEP] "+ self.cur_data_dict[index][k]
                        items.append(self.numericalize(input))

        # add vcpt
        if self.with_ts:
            cur_vis_sen = self.vcpt_dict[cur_vid_name][cur_start:cur_end + 1]
            cur_vscene_sen = self.vscene_dict[cur_vid_name][cur_start:cur_end+1]
            # ts = list(range(cur_start, cur_end + 1))
            # normalized_ts = self.normalize_tstamps(ts)
            cur_vis_sen = ", ".join(cur_vis_sen)
            ques = self.cur_data_dict[index]['q']

            q1 = ques[:split_idx]
            q2 = ques[split_idx:]

            q1_vcpt = self.numericalize_vcpt(cur_vis_sen, q1)
            q2_vcpt = self.numericalize_vcpt(cur_vis_sen, q2)
            q_vcpt = self.numericalize_vcpt(cur_vis_sen, q)

            items.append(q1_vcpt)
            items.append(q2_vcpt)
            items.append(q_vcpt)
            # items.append(vcpt_ts)
            # vscene, vscene_ts = self.numericalize_vcpts_with_ts(cur_vscene_sen, normalized_ts)
            # items.append(vscene)
            # items.append(vscene_ts)
        else:
            cur_vis_sen = self.vcpt_dict[cur_vid_name]
            cur_vscene_sen = self.vscene_dict[cur_vid_name]
            cur_vis_sen = " , ".join(cur_vis_sen)
            cur_vscene_sen = " , ".join(cur_vscene_sen)

            items.append(self.numericalize_vcpt(cur_vis_sen))
            items.append([1, 1]) #dummy value

            items.append(self.numericalize_vcpt(cur_vscene_sen))
            items.append([1, 1])  # dummy value


        # add other keys
        if self.mode == 'test':
            items.append(666)  # this value will not be used
        else:
            items.append(int(self.cur_data_dict[index][self.label_key]))
        for k in [self.qid_key]:
            items.append(self.cur_data_dict[index][k])
        items.append(cur_vid_name)

        # add visual feature
        if self.vfeat_load and self.vfeat not in ['i3d','c3d']:
            if self.with_ts:
                vid_feat = self.vid_h5[cur_vid_name][cur_start:cur_end]
                if len(vid_feat) == 0:
                    cur_vid_feat = torch.from_numpy(self.vid_h5[cur_vid_name][0:])
                else:
                    cur_vid_feat = torch.from_numpy(self.vid_h5[cur_vid_name][cur_start:cur_end])
            else:  # handled by vid_path
                cur_vid_feat = torch.from_numpy(self.vid_h5[cur_vid_name][:480])
            if self.normalize_v:
                cur_vid_feat = nn.functional.normalize(cur_vid_feat, p=2, dim=1)

        elif self.vfeat_load and self.vfeat == 'c3d':
            show = get_show_name(cur_vid_name)
            path = os.path.join(self.vid_feat_path, '{}_frames'.format(show), cur_vid_name,
                                '{}_{}_{}_{}.npy'.format(cur_vid_name, cur_vid_name, cur_start, cur_end))
            # cur_files = os.listdir(path)
            # if len(cur_files) > 1:
            #     localized_feats = []
            #     for cur_file in cur_files:
            #         localized_feats.append(torch.from_numpy(np.load(os.path.join(path, cur_file))))
            #     cur_vid_feat = torch.stack(localized_feats)
            # else:
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
                assert len(chunks_list) != 0, "chunks_list is zero for {} with ts:{},{}".format(cur_vid_name, cur_start, cur_end)

                if len(chunks_list) > 15 and self.vfeat_type == "conv":
                    cur_vid_feat = cur_vid_feat[chunks_list[:12]]
                else:
                    cur_vid_feat = cur_vid_feat[chunks_list]  # localized features

                # chunk, cur_vid_feat = self.get_features_for_fixed_blks(chunks_list, cur_vid_feat)
            else:
                ts = [chunks_list[0], chunks_list[-1]] # ts in terms of clips
                # num_blocks are larger here: 14 avg length
                chunk = get_random_chunk_from_ts(ts, self.num_blocks, len(cur_vid_feat))
                #assert len(chunk) == self.num_blocks, "length of chunk: {} should equals num_blocks{}".format(chunk, self.num_blocks)
                cur_vid_feat = cur_vid_feat[chunk[0]:chunk[-1]+1]
            if self.normalize_v:
                cur_vid_feat = nn.functional.normalize(cur_vid_feat, p=2, dim=1)
        else:
            cur_vid_feat = torch.zeros([2, 2])  # dummy placeholder
        items.append(cur_vid_feat)
        items.append(self.bow[cur_vid_name].values())
        # items.append(normalized_ts)
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
        # sentence_indices = self.tokenizer.encode(sentence, add_special_tokens=True)
        return sentence_indices

    def numericalize_qa(self, sen1, sen2, split_idx, eos=True):
        """convert words to indices"""
        # sentence_indices = [self.word2idx[w] if w in self.word2idx else self.word2idx["<unk>"]
        #                     for w in self.line_to_words(sentence, eos=eos)]  # 1 is <unk>, unknown
        q = self.line_to_words(sen1)
        q1 = q[:split_idx]
        q2 = q[split_idx:]
        qa = sen1 + " [SEP] "+ sen2
        q1_a = " ".join(q1) + " [SEP] " + sen2
        q2_a = " ".join(q2) + " [SEP] " + sen2
        qa_indices = self.tokenizer.encode(qa, add_special_tokens=True)
        q1a_indices = self.tokenizer.encode(q1_a, add_special_tokens=True)
        q2a_indices = self.tokenizer.encode(q2_a, add_special_tokens=True)
        return qa_indices, q1a_indices, q2a_indices

    def numericalize_vcpt(self, vcpt_sentence, q):
        """convert words to indices, additionally removes duplicated attr-object pairs"""
        attr_obj_pairs = vcpt_sentence.lower().split(",")  # comma is also removed
        unique_pairs = []
        for pair in attr_obj_pairs:
            pair = pair.strip()
            if pair not in unique_pairs:
                unique_pairs.append(pair)
        words = []
        for pair in unique_pairs:
            words.extend(pair.split())
        words.append("<eos>")

        # sentence_indices = [self.word2idx[w] if w in self.word2idx else self.word2idx["<unk>"]
        #                     for w in words]
        # print(q + " [SEP] "+ " [SEP] ".join(words))
        exit()
        sentence_indices = self.tokenizer.encode(q + " [SEP] "+ " [SEP] ".join(words), add_special_tokens=True)
        return sentence_indices

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


class Batch(object):
    def __init__(self):
        self.__doc__ = "empty initialization"

    @classmethod
    def get_batch(cls, keys=None, values=None):
        """Create a Batch directly from a number of Variables."""
        batch = cls()
        assert keys is not None and values is not None
        for k, v in zip(keys, values):
            setattr(batch, k, v)
        len_ = len(getattr(batch, 'answer_idx'))
        setattr(batch, 'batch_len', len_)
        return batch


def pad_collate(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq)."""
    def pad_sequences(sequences, key):
        if key in ["vcpt_ts", "vscene_ts"]:
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
    column_data = list(zip(*data))
    # text_keys = ["q", "qm1", "qm2", "a0", "a1", "a2", "a3", "a4", "sub", "vcpt", "vcpt_ts", "vscene", "vscene_ts"]
    text_keys = ["q1_a0", "q2_a0", "q1_a1", "q2_a1", "q1_a2", "q2_a2", "q1_a3", "q2_a3", "q1_a4", "q2_a4", "q_a0",
                 "q_a1", "q_a2", "q_a3", "q_a4", "sub", "q1_vcpt", "q2_vcpt", "vcpt"]
    label_key = "answer_idx"
    qid_key = "qid"
    vid_name_key = "vid_name"
    vid_feat_key = "vid"
    vfeat_type = "fc"
    all_keys = text_keys + [label_key, qid_key, vid_name_key, vid_feat_key]
    all_values = []
    for i, k in enumerate(all_keys):
        if k in text_keys:
            all_values.append(pad_sequences(column_data[i], k))
        elif k in [label_key]:
            # try:
            all_values.append(torch.LongTensor(column_data[i]))
            # except:
            #     all_values.append(torch.LongTensor([bow[:1331] for bow in column_data[i]]))
        elif k == vid_feat_key:
            if vfeat_type != 'conv':
                all_values.append(pad_video_sequences(column_data[i]))
            else:
                all_values.append(pad_video_tensors(column_data[i]))  # appends padded videos, lengths, and binary masks for lengths
        else:
            all_values.append(column_data[i])

    batched_data = Batch.get_batch(keys=all_keys, values=all_values)
    return batched_data


def preprocess_inputs(batched_data, max_sub_l, max_vcpt_l, max_vid_l, device="cuda:0"):
    """clip and move to target device"""
    max_len_dict = {"sub": max_sub_l, "vcpt": max_vcpt_l, "vcpt_ts":max_vcpt_l, "vid": max_vid_l, "q1_vcpt":max_vcpt_l, "q2_vcpt":max_vcpt_l}
    text_keys = ["q1_a0", "q2_a0", "q1_a1", "q2_a1", "q1_a2", "q2_a2", "q1_a3", "q2_a3", "q1_a4", "q2_a4", "q_a0",
                 "q_a1", "q_a2", "q_a3", "q_a4", "sub",  "q1_vcpt", "q2_vcpt", "vcpt"]
    label_key = "answer_idx"
    bow_key = 'bow'
    qid_key = "qid"
    vid_feat_key = "vid"
    # vfeat_type = "conv"
    model_in_list = []
    for k in text_keys + [vid_feat_key]:
        v = getattr(batched_data, k)
        if k in max_len_dict:
            if k == vid_feat_key:
                ctx, ctx_l, ctx_masks = v
                model_in_list.extend([ctx.to(device), ctx_l.to(device), ctx_masks.to(device)])
            else:
                ctx, ctx_l = v
                max_l = min(ctx.size(1), max_len_dict[k])
                if ctx.size(1) > max_l:
                    ctx_l = ctx_l.clamp(min=1, max=max_l)
                    ctx = ctx[:, :max_l]

                model_in_list.extend([ctx.to(device), ctx_l.to(device)])
        else:

             model_in_list.extend([v[0].to(device), v[1].to(device)])
    target_data = getattr(batched_data, label_key)
    # bow_target = getattr(batched_data, bow_key)
    # print(bow_target)
    target_data = target_data.to(device)
    # bow_target = bow_target.to(device)
    qid_data = getattr(batched_data, qid_key)
    return model_in_list, target_data, qid_data


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



__author__ = "Jie Lei"

import argparse
import os
import time

import torch

from utils import load_json, save_json_pretty


class BaseOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.opt = None

    def initialize(self):
        self.parser.add_argument("--debug", action="store_true", help="debug mode, break all loops")
        self.parser.add_argument("--results_dir_base", type=str, default="results/results")
        self.parser.add_argument("--log_freq", type=int, default=800, help="print, save training info")
        self.parser.add_argument("--log_train_freq", type=int, default=100, help="print, save only training info")
        self.parser.add_argument("--feat_attn_merge_op", type=str, default="sum", help="In MAC unit, `sum`: weighted sum of features, `mean`: weighted mean of features.")
        self.parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
        self.parser.add_argument("--margin", type=float, default=0.6, help="Margin for multimargin loss")
        self.parser.add_argument("--wd", type=float, default=1e-5, help="weight decay")
        self.parser.add_argument("--n_epoch", type=int, default=20, help="number of epochs to run")
        self.parser.add_argument("--random_seed", type=int, default=2018, help="random seed for weight init. Set randomly in code.")
        self.parser.add_argument("--max_es_cnt", type=int, default=5, help="number of epochs to early stop")
        self.parser.add_argument("--bsz", type=int, default=16, help="mini-batch size")
        self.parser.add_argument("--test_bsz", type=int, default=16, help="mini-batch size for testing")
        self.parser.add_argument('--clip', type=float, default=0.0, help='gradient clipping, 0.0: for no gradient clipping; any value > 0 for clipping')
        self.parser.add_argument("--wt1", type=float, default=1.0, help="weight hyperparameter for cross entropy loss")
        self.parser.add_argument("--wt2", type=float, default=1.0, help="weight hyperparameter for separate stream loss")
        self.parser.add_argument("--wt3", type=float, default=0.0, help="weight hyperparameter for cross attn loss")
        self.parser.add_argument("--wt4", type=float, default=0.0, help="weight hyperparameter for vid stream loss")
        self.parser.add_argument("--lambda_", type=float, default=0.1, help="l1 regularization")
        self.parser.add_argument("--bowsz", type=int, default=1331, help="Size of bow vocab")
        self.parser.add_argument("--no_log", action="store_false", help="Not use log in reasoning loss function")
        self.parser.add_argument("--device", type=int, default=0, help="gpu ordinal, -1 indicates cpu")
        self.parser.add_argument("--multi_gpu", type=int, default=1, help="multi-gpu support flag")
        self.parser.add_argument("--no_core_driver", action="store_true",
                                 help="hdf5 driver, default use `core` (load into RAM), if specified, use `None`")
        self.parser.add_argument("--pred_slct_criterion", type=str, default="best", choices=["best", "avg"],
                                 help="way to combine predictions from classifier and reasoning graph")
        self.parser.add_argument("--word_count_threshold", type=int, default=2, help="word vocabulary threshold")
        self.parser.add_argument("--resume_epoch", type=int, default=-1, help="Resume training from resume_epoch, if -1, start from zero")

        # model config
        self.parser.add_argument("--no_glove", action="store_false", help="not use glove vectors")
        self.parser.add_argument("--no_ts", action="store_true", help="no timestep annotation, use full length feature")
        self.parser.add_argument("--no_reverse", action="store_true", help="reverse input order")
        self.parser.add_argument("--vfeat", type=str, default='i3d', help="feature type, i3d: i3d fc features, c3d: c3d fc features")
        self.parser.add_argument("--vfeat_type", type=str, default='fc', help="feature type, fc: i3d fc features, conv: i3d conv features")
        self.parser.add_argument("--input_streams", type=str, nargs="+", choices=["vcpt", "sub", "imagenet","set_attn", "q_only"],
                                 help="input streams for the model, will use both `vcpt` and `sub` streams")
        self.parser.add_argument("--model_config", type=int, default=1, choices=[1,2],
                                 help="model configs: 1: model w/ 2 losses, 2: model w/1 loss w/o downsizing features")

        self.parser.add_argument("--n_layers_cls", type=int, default=2, help="number of layers in classifier")
        self.parser.add_argument("--n_layers_stacked", type=int, default=1, help="number of layers in classifier")
        self.parser.add_argument("--n_heads", type=int, default=1, help="number of layers in classifier")
        self.parser.add_argument("--add_src_vec", type=int, default=0, help="number of layers in classifier")
        self.parser.add_argument("--src_vec_dim", type=int, default=32, help="number of layers in classifier")
        self.parser.add_argument("--add_src_embed", type=int, default=0, help="use embedding for src")
        self.parser.add_argument("--freeze_bert", type=int, default=0,
                                                 help="1=freeze berts, 0: trainable bert")
        self.parser.add_argument("--pretrained_bert", type=int, default=1,
                                                         help="1=initialize w. pre-trained bert, 0: train from scratch")
        self.parser.add_argument("--add_verbs_nouns", type=int, default=0,
                                                                 help="1=add verbs and nouns to input, 0: don't add")
        self.parser.add_argument("--num_masks", type=int, default=2,
                                 help="number of masks to split question into. Possible values: {2, 3, 4}")
        self.parser.add_argument("--num_blocks", type=int, default=4,
                                 help="number of feat. blocks; num_blocks == num_units in modified mac cell")
        self.parser.add_argument("--mac_steps", type=int, default=2,
                                 help="number of reasoning steps in MACUnit")
        self.parser.add_argument("--mac_version", type=str, default="simple",
                                 help="different mac variants.'vanilla': for original mac, 'simple': for modified version ")
        self.parser.add_argument("--hsz1", type=int, default=150, help="hidden size for the first lstm")
        self.parser.add_argument("--hsz2", type=int, default=300, help="time expansion dim size")
        self.parser.add_argument("--tsz", type=int, default=16, help="hidden size for the first lstm")
        self.parser.add_argument("--embedding_size", type=int, default=768, help="word embedding dim")
        self.parser.add_argument("--max_sub_l", type=int, default=1, help="max length for subtitle")
        self.parser.add_argument("--max_vcpt_l", type=int, default=1, help="max length for visual concepts")
        self.parser.add_argument("--max_cap_l", type=int, default=128, help="max length for dense captions")
        self.parser.add_argument("--max_seq_len", type=int, default=256, help="max length for BERT input")
        self.parser.add_argument("--max_vid_l", type=int, default=1, help="max length for video feature")
        self.parser.add_argument("--max_total_l", type=int, default=1, help="max length for video feature")
        self.parser.add_argument("--vocab_size", type=int, default=0, help="vocabulary size")
        self.parser.add_argument("--aux_loss", type=int, default=0, help="auxiliary loss for reasoning module")
        self.parser.add_argument("--no_normalize_v", action="store_false", help="do not normalize video featrue")
        self.parser.add_argument("--position_aware_knowledge", default=1, help="position aware knowledge in read unit")
        self.parser.add_argument("--model_opt", default=0, help="position aware knowledge in read unit")
        self.parser.add_argument("--fusion_config", default=3, help="0: lstm-level, 1: attn level, 2: before classifier, 4: late fusion")
        self.parser.add_argument("--word_drop", default=0, help="whether or not use word dropout")
        self.parser.add_argument("--is_flip", default=0, help="whether or not flip the question")
        self.parser.add_argument("--add_caption", default=0, help="whether or not use caption")
        self.parser.add_argument("--question_group", default='where', help="Type of questions, we want to train/test network on."
                                                                      "choices='what','who','where','why','how','others' ")

        #baseline config
        self.parser.add_argument("--baseline", default=1, help="1: concat q and answer features, 2: concat q and "
                                                               "similarity scores of q with answers")
        self.parser.add_argument("--train_baseline", type=bool, default=False, help="Set True if want to train baseline models.")

        # path config
        self.parser.add_argument("--train_path", type=str, default="data/tvqa_train_processed.json",
                                 help="train set path, tvqa_grouped_train_data.json for qa groups")
        self.parser.add_argument("--valid_path", type=str, default="data/tvqa_val_processed.json",
                                 help="valid set path, tvqa_grouped_val_data.json for qa groups")
        self.parser.add_argument("--test_path", type=str, default="data/visual_only_questions_clean.json",
                                 help="test set path")
        self.parser.add_argument("--glove_path", type=str, default="data/glove.6B.300d.txt",
                                 help="GloVe pretrained vector path")
        self.parser.add_argument("--fasttext_path", type=str, default="data/crawl-300d-2M-subword.vec",
                                 help="GloVe pretrained vector path")
        self.parser.add_argument("--bert_path", type=str,
                                 default="../../cache/bert_embedding.pickle",
                                 help="Bert pretrained vector path")
        self.parser.add_argument("--vcpt_path", type=str, default="data/det_visual_concepts_hq.pickle",
                                 help="visual concepts feature path")
        self.parser.add_argument("--vscene_path", type=str, default="data/det_visual_scene_attributes_clean_hq.pickle",
                                 help="visual scene concepts feature path")
        self.parser.add_argument("--bow_path", type=str, default="data/vcpt_bow.json", help="path for bow vectors")
        self.parser.add_argument("--vid_feat_path", type=str, default="data/i3d_features_fc_resize",
                                 help="imagenet feature path")
        self.parser.add_argument("--vid_feat_size", type=int, default=1024,
                                 help="visual feature dimension, 1024 for i3d fc feat, 4096 for c3d fc feats, 2048 for imagenet")
        self.parser.add_argument("--word2idx_path", type=str, default="cache/word2idx.pickle",
                                 help="word2idx cache path")
        self.parser.add_argument("--idx2word_path", type=str, default="cache/idx2word.pickle",
                                 help="idx2word cache path")
        self.parser.add_argument("--vocab_embedding_path", type=str, default="cache/vocab_embedding.pickle",
                                 help="vocab_embedding cache path")
        self.parser.add_argument("--vocab_embedding_path2", type=str, default="cache/fasttext_vocab_embedding.pickle",
                                 help="vocab_embedding cache path for fasttext")

        self.parser.add_argument("--captions_path", type=str, default="data/full_caps_data/",
                                 help="path to dense captions folder")
        #evaluation config
        self.parser.add_argument("--topk", type=int, default=1, help="#reasoning paths to be considered for evaluation")
        self.parser.add_argument("--eval_type", type=str, default='majority', help="how to compute loss term2, "
                                                                                   "'majority': do majority voting for pred."
                                                                                   "'all' : do pred on all in prob_matrix ")
        self.initialized = True

    def display_save(self, options, results_dir):
        """save config info for future reference, and print"""
        args = vars(options)  # type == dict
        # Display settings
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # Save settings
        if not isinstance(self, TestOptions):
            option_file_path = os.path.join(results_dir, 'opt.json')  # not yaml file indeed
            save_json_pretty(args, option_file_path)

    def parse(self):
        """parse cmd line arguments and do some preprocessing"""
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()
        if opt.resume_epoch > -1:
            results_dir = 'results_2019_07_22_18_26_03'
        else:
            results_dir = opt.results_dir_base + time.strftime("_%Y_%m_%d_%H_%M_%S")

        if isinstance(self, TestOptions):
            options = load_json(os.path.join("results", opt.model_dir, "opt.json"))
            for arg in options:
                setattr(opt, arg, options[arg])
        else:
            if opt.resume_epoch == -1:
                os.makedirs(results_dir)
                self.display_save(opt, results_dir)

        opt.normalize_v = not opt.no_normalize_v
        opt.device = torch.device("cuda:%d"%opt.device if opt.device is not -1 else "cpu")
        opt.with_ts = not opt.no_ts
        opt.with_log = not opt.no_log
        opt.input_streams = [] if opt.input_streams is None else opt.input_streams
        opt.vid_feat_flag = True if "imagenet" in opt.input_streams else False
        opt.h5driver = None if opt.no_core_driver else "core"
        opt.results_dir = results_dir

        self.opt = opt
        return opt


class TestOptions(BaseOptions):
    """add additional options for evaluating"""
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument("--model_dir", type=str, help="dir contains the model file")
        self.parser.add_argument("--model_dir2", type=str, help="dir contains the model file")
        self.parser.add_argument("--ensemble_dir", type=str, help="dir contains the model file")
        self.parser.add_argument("--mode", type=str, default="valid", help="valid/test")


if __name__ == "__main__":
    import sys
    sys.argv[1:] = ["--input_streams", "vcpt"]
    opt = BaseOptions().parse()


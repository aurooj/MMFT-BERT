#from main_ensemble_stagewise import Ensemble
#from model.optimal_reasoning import OptimalReasoning

__author__ = "Jie Lei"

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
import torch
import torch.nn as nn
import csv
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from tvqa_vqa_2bert_bertfusion import ABC
from tvqa_dataset_vqa_bert_attn import TVQADataset, pad_collate, preprocess_inputs
from config import TestOptions
from utils import merge_two_dicts, save_json, select_best_preds, load_json, load_pickle

def get_split_index(list_to_find):
    if 7595 in list_to_find:
        return list_to_find.index(7595)
    elif 21437 in list_to_find:
        return list_to_find.index(21437)
    elif 4770 in list_to_find:
        return list_to_find.index(4770)
    elif 11923 in list_to_find:
        return list_to_find.index(11923)

def test(opt, dset, model):
    max_len_dict = {"sub": opt.max_sub_l,
                    "vcpt": opt.max_vcpt_l,
                    "vscene": opt.max_vcpt_l,
                    "vcpt_ts": opt.max_vcpt_l,
                    "vscene_ts": opt.max_vcpt_l,
                    "vid": opt.max_vid_l,
                    "max_seq_len":opt.max_seq_len}
    dset.set_mode(opt.mode)
    torch.set_grad_enabled(False)
    model.eval()
    valid_loader = DataLoader(dset, batch_size=opt.test_bsz, shuffle=False, collate_fn=pad_collate)

    qid2preds = {}
    qid2targets = {}
    qid2predscores = {}
    qid2attn = {}
    count = 0
    for valid_idx, batch in tqdm(enumerate(valid_loader)):
#       print(opt.device)
       model_inputs, targets, qids = preprocess_inputs(batch, max_len_dict, device=opt.device)
       outputs, attn = model(**model_inputs)
#       print(outputs['joint_scores'])
       if isinstance(outputs, dict):
           out = outputs['joint_scores']
       else:
           out = outputs[2]
       pred_ids = out.data.max(1)[1].cpu().numpy().tolist()
       cur_qid2preds = {qid: pred for qid, pred in zip(qids, pred_ids)}
       qid2preds = merge_two_dicts(qid2preds, cur_qid2preds)
       cur_qid2predscores = {qid: scores.data.cpu().numpy().tolist() for qid, scores in zip(qids, out)}
       #cur_qid2attn = {qid: attn_.data.cpu().numpy().tolist() for qid, attn_ in zip(qids, attn)}

       #qid2attn = merge_two_dicts(qid2attn, cur_qid2attn)
       qid2predscores = merge_two_dicts(qid2predscores, cur_qid2predscores)
       cur_qid2targets = {qid:  target for qid, target in zip(qids, targets.cpu().numpy().tolist())}
       qid2targets = merge_two_dicts(qid2targets, cur_qid2targets)
    # print("#unflipped questions:{}".format(count))
    return qid2preds, qid2targets, qid2predscores, None


def get_acc_from_qid_dicts(qid2preds, qid2targets):
    qids = qid2preds.keys()
    preds = np.asarray([int(qid2preds[ele]) for ele in qids]) #if dictionary has tuple items
    targets = np.asarray([int(qid2targets[ele]) for ele in qids])
    acc = sum(preds == targets) / float(len(preds))
    return acc

def interpret_predictions(opt):
    raw_valid = load_json(opt.valid_path)
    pred_path = os.path.join("results", opt.model_dir, "qid2preds_%s.json" % opt.mode)
    qid2preds = load_json(pred_path)

    with open('results/{}/interpreted_results.csv'.format(opt.model_dir), mode='w') as csv_file:
        fieldnames = ['clip_name', 'qid','q_family', 'q', 'ans_choices', 'correct_ans', 'pred_ans', 'probs']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for item in raw_valid:
            qid = item['qid']
            try:
                pred_a_id = qid2preds[str(qid)][0]
                lbl_a_id = item['answer_idx']
                if pred_a_id != lbl_a_id:
                    q = item['q']
                    q_family = q.split(' ')[0]
                    correct_ans = item["a{}".format(lbl_a_id)]
                    pred_ans = item["a{}".format(pred_a_id)]
                    probs = qid2preds[str(qid)][1]

                    a0, a1, a2, a3, a4 = item['a0'], item['a1'], item['a2'], item['a3'], item['a4']

                    writer.writerow({'clip_name': item['vid_name'] ,'qid': qid, 'q_family':q_family ,'q': q,
                                     'ans_choices': "a0{}\na1{}\na2{}\na3{}\na4{}".format(a0,a1,a2,a3,a4),
                                     'correct_ans':correct_ans, 'pred_ans':pred_ans, 'probs':probs})

            except:
                continue

def get_questionwise_accuracy(opt, qid2preds):
    raw_valid = load_json(opt.valid_path)
    #pred_path = os.path.join("results", opt.model_dir, "qid2preds_%s_with_probs.json" % opt.mode)
    #qid2preds = load_json(pred_path)
    mine_family_dic = {'what': 0, 'who': 0, 'where': 0, 'why': 0, 'how': 0, 'others': 0}
    # their_family_dic = {'what': 0, 'who': 0, 'where': 0, 'why': 0, 'how': 0, 'others': 0}
    fam_count = {'what': 0, 'who': 0, 'where': 0, 'why': 0, 'how': 0, 'others': 0}
    with open('results/{}/questionwise_acc.csv'.format(opt.model_dir), mode='w') as csv_file:
        # fieldnames = ['clip_name', 'qid', 'q_family', 'q', 'ans_choices', 'correct_ans', 'mynet_pred', 'mynet_prob', 'tvqanet_pred', 'tvqa_prob']
        family_fields = ['what','who','where','why','how','others']
        writer = csv.DictWriter(csv_file, fieldnames=family_fields)
        writer.writeheader()

        for idx, item in tqdm(enumerate(raw_valid)):
            qid = str(item['qid'])
            pred_id = qid2preds[qid]
            lbl_id = item['answer_idx']
            q = item['q']
            q_family = q.split(' ')[0]
            q_family = q_family.lower()

            is_correct = pred_id == lbl_id

            if is_correct:
                if q_family in ['what','who','where','why','how']:
                        mine_family_dic[q_family] += 1
                        fam_count[q_family] += 1
                else:
                    mine_family_dic['others'] +=1
                    fam_count['others'] += 1

        writer.writerow({'what': fam_count['what'], 'who': fam_count['who'],
                         'where': fam_count['where'], 'why': fam_count['why'],
                         'how': fam_count['how'], 'others': fam_count['others']})
        writer.writerow({'what': mine_family_dic['what'], 'who': mine_family_dic['who'],
                         'where': mine_family_dic['where'], 'why': mine_family_dic['why'],
                         'how': mine_family_dic['how'], 'others': mine_family_dic['others']})
        print(mine_family_dic)
        print(fam_count)

if __name__ == "__main__":
    opt = TestOptions().parse()
    #opt.test_path = "../../data/tvqa_test_public_processed.json"
    #opt.valid_path = "../../data/tvqa_val_processed.json"
    #opt.train_path = "../../data/tvqa_train_processed.json"
    #opt.glove_path = "../../data/glove.6B.300d.txt"
    #opt.vcpt_path = "../../data/det_visual_concepts_hq.pickle"
    #opt.vscene_path ="../../data/det_visual_scene_attributes_clean_hq.pickle"
    #opt.bow_path="../../data/vcpt_bow.json"
    #opt.word2idx_path = "../../cache/word2idx.pickle"
    #opt.idx2word_path = "../../cache/idx2word.pickle"
    #opt.vocab_embedding_path = "../../cache/vocab_embedding.pickle"
    # opt.test_bsz = 1
    #opt.vid_feat_path = "data/i3d_features_fc_resize"
    # opt.test_bsz = 1
    # opt.aux_loss = 0
    #opt.mode = 'test'
    dset = TVQADataset(opt)
    print(len(dset))
    opt.vocab_size = len(dset.word2idx)
    # opt.model_opt = 0
    #opt.model_dir = 'results_2020_05_26_06_00_51'
    # opt.model_dir2 = 'results_2019_05_13_17_52_36'
    #
    # model = Ensemble(opt)
    model = ABC(opt)
    # #
    model.to(opt.device)
    if torch.cuda.device_count() > 1 and opt.multi_gpu:
        print("Let's use {} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
        
    cudnn.benchmark = True
    # # model_path = os.path.join("results", opt.ensemble_dir, "best_valid.pth")
    model_path = os.path.join("results", opt.model_dir, "best_valid.pth")
    model.load_state_dict(torch.load(model_path), strict=False)
    # # #
    all_qid2preds, all_qid2targets, all_qid2predscores, all_qid2attn = test(opt, dset, model)
    
    # #
    if opt.mode == "valid":
         accuracy = get_acc_from_qid_dicts(all_qid2preds, all_qid2targets)
         print("In valid mode, accuracy is %.10f" % accuracy)
    # #
    results = {'qid2scores': all_qid2predscores,
               'qid2targets': all_qid2targets,
               'qid2preds': all_qid2preds,
               'qid2attn': all_qid2attn}

    save_path = "data/pred_scores_{}_".format(opt.mode)+opt.model_dir+".json"

    save_json(results, save_path)
    

    save_json(all_qid2preds, 'data/prediction_{}_'.format(opt.mode)+opt.model_dir+'.json')
    #if opt.mode == "valid":
    #    all_qid2preds = load_json('../../data/prediction_valid_results_2020_05_28_20_21_31.json')
    #print(get_acc_from_qid_dicts(all_qid2preds, all_qid2targets))
    #    get_questionwise_accuracy(opt, all_qid2preds)


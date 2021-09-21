import csv

__author__ = "Jie Lei"

import os
import sys
import re
import math
import json
import glob
import copy
import pysrt
import numpy as np
import shutil
import matplotlib.pyplot as plt
from collections import OrderedDict

from tqdm import tqdm

from utils import read_json_lines, load_json, save_json, get_show_name, load_pickle, save_pickle


def merge_list_dicts(list_dicts):
    z = list_dicts[0].copy()   # start with x's keys and values
    for i in range(1, len(list_dicts)):
        z.update(list_dicts[i])  # modifies z with y's keys and values & returns None
    return z


def get_vidname2cnt_per_show(base_path):
    """ get jpg file count for each sub dirs in the base_path
    the resulting file is a python dict with {subdir_name: count}
    """
    subdirs = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]
    vidname2cnt = {}
    for ele in tqdm(subdirs):
        cur_subdir_path = os.path.join(base_path, ele)
        # cur_files = [name for name in os.listdir(cur_subdir_path) 
        #              if os.path.isfile(os.path.join(cur_subdir_path, name))]
        cur_files = glob.glob(os.path.join(cur_subdir_path, "*jpg"))
        vidname2cnt[ele] = len(cur_files)
    return vidname2cnt


def get_vidname2cnt_all(frame_root_path, vidname2cnt_cache_path):
    if os.path.exists(vidname2cnt_cache_path):
        print("Found frame cnt cache, loading ...")
        return load_json(vidname2cnt_cache_path)
    show_names = ["bbt", "friends", "grey", "met", "castle", "house"]
    vidname2cnt_list = []
    for sn in show_names:
        print("Count frames in %s" % sn)
        cur_base_path = os.path.join(frame_root_path, "%s_frames" % sn)
        vidname2cnt_list.append(get_vidname2cnt_per_show(cur_base_path))
    vidname2cnt = merge_list_dicts(vidname2cnt_list)    
    save_json(vidname2cnt, vidname2cnt_cache_path)    
    return 


def clean_str(string):
    """ Tokenization/string cleaning for strings.
    Taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?:.\'`]", " ", string)  # <> are added after the cleaning process
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)  # split as two words
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r":", " : ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\.\.\.", " . ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def load_srt(srt_dir, srt_cache_path):
    """
    return: A python dict, the keys are the video names, the entries are lists,
            each contains all the text from a .srt file
    sub_times are the start time of the sentences.
    """
    if os.path.exists(srt_cache_path):
        print("Found srt data cache, loading ...")
        return load_json(srt_cache_path)

    print("Loading srt files from %s ..." % srt_dir)
    srt_paths = glob.glob(os.path.join(srt_dir, "*.srt"))
    name2sub_text = {}
    name2sub_time = {}
    for i in tqdm(range(len(srt_paths))):
        subs = pysrt.open(srt_paths[i], encoding="iso-8859-1")
        if len(subs) == 0:
            subs = pysrt.open(srt_paths[i])

        text_list = []
        sub_time_list = []
        for j in range(len(subs)):
            cur_sub = subs[j]
            cur_str = cur_sub.text
            cur_str = "(<UNKNAME>:)" + cur_str if cur_str[0] != "(" else cur_str
            cur_str = cur_str.replace("\n", " ")
            text_list.append(cur_str)
            sub_time_list.append(
                60 * cur_sub.start.minutes + cur_sub.start.seconds + 0.001 * cur_sub.start.milliseconds)

        key_str = os.path.splitext(os.path.basename(srt_paths[i]))[0]
        name2sub_text[key_str] = text_list
        name2sub_time[key_str] = sub_time_list
    srt_data = {"sub_text": name2sub_text, "sub_time": name2sub_time}
    save_json(srt_data, srt_cache_path)
    return srt_data


def convert_ts(ts):
    """ 26.2-34.4  -->  [26.2, 34.4] ,
    also replace any NaN value with [10, 30], a simple replacement, will fix later"""
    new_ts = [float(ele) for ele in ts.split("-")]
    is_nan = False
    if math.isnan(new_ts[0]) or math.isnan(new_ts[1]):
        new_ts = [10, 30]  #
        is_nan = True
    return new_ts, is_nan


def interval2frame(interval, num_frame, fps=3):
    """ downsample to 300 frame max,
    :param interval: e.g. [26.2, 34.4]
    :param num_frame: number of frame for this clip
    :param fps: number of frames used per second
    :return:
    """
    # 0.0356 of the video has more than 300 frames, for those, downsample to 300.
    max_num_frame = 300.
    if num_frame > max_num_frame:
        frame_start_end = [(max_num_frame / num_frame) * fps * ele for ele in interval]
    else:
        frame_start_end = [fps * ele for ele in interval]

    frame_start_end = np.asarray([frame_start_end[0] - fps, frame_start_end[1] + fps])
    frame_start_end = np.floor(np.clip(frame_start_end, 0, 300))
    if frame_start_end[0] == frame_start_end[1]:
        frame_start_end[0] = max(0, frame_start_end[0] - 3)
    frame_start_end = [int(x) for x in frame_start_end]
    return frame_start_end


def tokenize_qa(data_dicts):
    """tokenize the text in QAs"""
    tokenized_data_dicts = []
    text_keys = ["q", "a0", "a1", "a2", "a3", "a4"]
    all_keys = data_dicts[0].keys()
    print("Tokenize QA ...")
    for ele in tqdm(data_dicts):
        tmp_dict = {}
        for k in all_keys:
            if k in text_keys:
                tmp_dict[k] = clean_str(ele[k])
            else:
                tmp_dict[k] = ele[k]
        tokenized_data_dicts.append(tmp_dict)
    return tokenized_data_dicts


def tokenize_srt(srt_data):
    """tokenize the text in srt"""
    tokenized_srt_data = {"sub_text": {}, "sub_time": srt_data["sub_time"]}
    print("Tokenize subtitle ...")
    for k in tqdm(srt_data["sub_text"].keys()):
        tokenized_srt_data["sub_text"][k] = [clean_str(s) for s in srt_data["sub_text"][k]]
    return tokenized_srt_data


def add_srt(raw_data_dicts, srt_data, eos_token="<eos>"):
    """ add entries 'sub_time', 'sub_text' """
    data_dicts = copy.deepcopy(raw_data_dicts)
    eos_token = " %s " % eos_token  # add space around
    print("Adding subtitle ...")
    for i in tqdm(range(len(data_dicts))):
        vid_name = data_dicts[i]["vid_name"]
        data_dicts[i]["sub_text"] = eos_token.join(srt_data["sub_text"][vid_name])
        data_dicts[i]["sub_time"] = srt_data["sub_time"][vid_name]
    return data_dicts


def find_nearest(array, value):
    """closet value in an array to a given value"""
    idx = (np.abs(array-value)).argmin()
    return idx  # array[idx]


def get_located_sub_text(ts, sub_text_list, sub_time, eos_token="<eos>"):
    """return the located subtitle text according to the timestep annotation
    :param ts: (list) e.g. [26.2, 34.4]
    :param sub_text_list: (list) each element is a subtitle sentence
    :param sub_time: (list) each element is a float number indicates the start time of a subtitle sentence
    """
    located_indices = []
    for idx in range(len(sub_time)):
        if ts[0] < sub_time[idx] < ts[1]:
            located_indices.append(idx)

    # deal with 0-length: use three sub sentences most close to START
    if len(located_indices) == 0:
        closest_1 = find_nearest(np.asarray(sub_time), ts[0])
        located_indices.extend([closest_1 - 1, closest_1, closest_1 + 1])

    # rm the indices larger than length of sub_text_list or negative
    located_indices = [located_indices[i] for i in range(len(located_indices))
                       if located_indices[i] <= len(sub_text_list) - 1 and
                          located_indices[i] >= 0 ]

    # add the one before the first located ts, no need to do it for the last one
    if 0 not in located_indices:
        located_indices = [located_indices[0] - 1] + located_indices
    eos_token = " %s " % eos_token
    located_sub_text = eos_token.join([sub_text_list[idx] for idx in located_indices])
    return located_sub_text


def add_located(raw_data_dicts, srt_data, frame_cnt):
    """ add entries 'located_frame', 'located_sub_text' """
    data_dicts = copy.deepcopy(raw_data_dicts)
    nan_cnt = 0
    for i in tqdm(range(len(data_dicts))):
        vid_name = data_dicts[i]["vid_name"]
        sub_text_list = srt_data["sub_text"][vid_name]
        sub_time = srt_data["sub_time"][vid_name]
        ts, is_nan = convert_ts(data_dicts[i]["ts"])
        nan_cnt += is_nan
        data_dicts[i]["ts"] = ts
        data_dicts[i]["located_frame"] = interval2frame(ts, frame_cnt[vid_name])
        data_dicts[i]["located_sub_text"] = get_located_sub_text(ts, sub_text_list, sub_time)
    print("There are %d NaN values in ts, which are replaced by [10, 30], will be fixed later" % nan_cnt)
    return data_dicts


def process_qa(qa_path, processed_srt, frame_base_path, frame_cnt_cache_path, save_path):
    qa_data = read_json_lines(qa_path)
    qa_data = tokenize_qa(qa_data)
    qa_srt_data = add_srt(qa_data, processed_srt, eos_token="<eos>")
    frame_cnt_dict = get_vidname2cnt_all(frame_base_path, frame_cnt_cache_path)
    qa_srt_located_data = add_located(qa_srt_data, processed_srt, frame_cnt_dict)
    save_json(qa_srt_located_data, save_path)

########## my functions ############

def group_data(split='train'):
    raw_data = load_json("data/tvqa_{}_processed.json".format(split))
    grouped_data_dic = {'what':[], 'who':[], 'where':[], 'why':[], 'how':[], 'others':[]}

    for item in tqdm(raw_data):
        q = item['q']
        q_family = q.split(' ')[0].lower()
        if q_family in ['what','who','where','why','how']:
            grouped_data_dic[q_family].append(item)
        else:
            grouped_data_dic['others'].append(item)
    save_json(grouped_data_dic, 'data/grouped_{}_data.json'.format(split))


def get_visual_ques_stats():
    #todo:
    # read isolated qids csv
    # read validation set json
    # for each id in isolated qids, retreive all related information
    # sort info family-wise, tv-series wise, and put all info for isolated qids in one csv.
    # put all attention maps into one folder
    visual_only_qids = []
    qids = []
    save_dir = 'attention/visual_only_questions'
    vcpt_dir = 'attention/vcpt_attn'
    q_dir = 'attention/q_attn'
    ans_dir = 'attention/ans_similarities'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    raw_valid = load_json("./data/tvqa_val_processed.json")
    with open('data/visual_only_clean.csv') as csvfile:
        qidCSV = csv.reader(csvfile, delimiter=',')
        for row in qidCSV:
            print(row[0])
            qids.append(int(row[0]))
        print(qids)

    #filterout info for only those qids
    for item in raw_valid:
        if item['qid'] in qids:
            visual_only_qids.append(item)

    #write out info in a csv
    with open('data/visual_only_clean_correct.csv', mode='w') as csv_file:
        fieldnames = ['qid', 'clip_name', 'show', 'q_family', 'q', 'ans_choices', 'correct_ans']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for item in visual_only_qids:
            q = item['q']
            qid = str(item['qid'])
            q_family = q.split(' ')[0]
            lbl_a_id = item['answer_idx']
            correct_ans = item["a{}".format(lbl_a_id)]
            a0, a1, a2, a3, a4 = item['a0'], item['a1'], item['a2'], item['a3'], item['a4']
            writer.writerow({'qid': qid, 'clip_name': item['vid_name'], 'show':get_show_name(item['vid_name']),'q_family': q_family, 'q': q,
                             'ans_choices': "a0{}\na1{}\na2{}\na3{}\na4{}".format(a0, a1, a2, a3, a4),
                             'correct_ans': correct_ans})

            # #saving attentions maps into one folder
            # try:
            #     shutil.copy(os.path.join(vcpt_dir, qid+'_attn.jpg'), os.path.join(save_dir, qid+'_attn.jpg'))
            #     shutil.copy(os.path.join(q_dir, qid + '_attn.jpg'), os.path.join(save_dir, qid + '_qattn.jpg'))
            #     shutil.copy(os.path.join(ans_dir, qid + '_ans_sim.jpg'), os.path.join(save_dir, qid + '_ans_sim.jpg'))
            # except:
            #     continue

    save_json(visual_only_qids, 'data/visual_only_questions_clean.json')

def merge_visual_and_scene_cpts(vcpt_path, vscene_path):
    #this function merges two dictionaries with same keys by concatenating their string values for each key
    vcpt_dict = load_pickle(vcpt_path)
    scene_dict = load_pickle(vscene_path)

    scene_dict = clean_scene_lbls(scene_dict)
    d = {}
    for k in tqdm(vcpt_dict.keys()):
        d[k] = [",".join([m, n]) for m, n in zip(scene_dict[k], vcpt_dict[k])]

    save_pickle(d, 'data/det_visual_concepts_combined_reverse_order_hq.pickle')

def clean_scene_lbls(scene_dict):
    for k in tqdm(scene_dict.keys()):
        scene_dict[k] = [clean_str(str) for str in scene_dict[k]]

    return scene_dict

def filter_scene_lbls(scene_dict):
    #just keep indoor outdoor and top scene category
    for k in tqdm(scene_dict.keys()):
        scene_dict[k] = [split_at(clean_str(str), ',', 6)[1] for str in scene_dict[k]]

    save_pickle(scene_dict, './data/det_visual_scene_attributes_hq.pickle')

def split_at(s, delim, n):
    groups = s.split(delim)
    return ','.join(groups[:n]),','.join(groups[n:])

def plot_answer_words_freq():
    raw_train = load_json('./data/tvqa_train_processed.json')
    raw_valid = load_json('./data/tvqa_val_processed.json')
    data_list = raw_train
    data_list.extend(raw_valid)

    print('\n collecting all answers in one list..')
    answers = []
    for item in tqdm(data_list):
        answers.append(item['a'+str(item['answer_idx'])])
    import re
    print('\n cleaning answers...')
    for i in tqdm(range(len(answers))):
        answers[i] = clean_str(answers[i])

    import nltk
    # nltk.download('punkt')
    print('\n creating word freq dictionary. this may take few minutes...')
    wordfreq = {}
    for sentence in tqdm(answers):
        tokens = nltk.word_tokenize(sentence)
        for token in tokens:
            if token not in wordfreq.keys():
                wordfreq[token] = 1
            else:
                wordfreq[token] += 1

    save_json(wordfreq, './data/answers_word_frequency.json')
    wordfreq = load_json('./data/answers_word_frequency.json')
    print(len(wordfreq))
    # wordfreq = OrderedDict(sorted(wordfreq.items(), key=lambda x: x[1], reverse=True))
    wordfreq_short = wordfreq
    for k,v in wordfreq_short.items():
        if v < 5:
            del wordfreq_short[k]
    print(wordfreq_short)
    print('\n plotting word freq in answers...')
    plt.figure()
    plt.plot(wordfreq_short.keys(), wordfreq_short.values())
    plt.show()

def get_unique_object_list():
    vcpts = load_pickle('./data/det_visual_concepts_hq.pickle')
    unique_pairs = set()
    attributes, objects = [], []

    for k, v in tqdm(vcpts.items()):
        for item in v:
            unique_pairs.update(item.lower().split(","))

    for pair in tqdm(list(unique_pairs)):
        if len(pair.split()) > 1:
            if len(pair.split()) == 3:
                print(pair)
                attr, obj1, obj2 = pair.split()
                obj = "_".join([obj1, obj2])
            elif len(pair.split()) == 2:
                attr, obj = pair.split()
            else:
                parts = pair.split()
                attr = "_".join(parts[:2])
                obj = "_".join(parts[2:])
                print(pair)
            if attr not in attributes:
                attributes.append(attr.strip())
            if obj not in objects:
                objects.append(obj.strip())
        else:
            if pair not in objects:
                objects.append(pair.strip())
    save_json(attributes, "./data/unique_attr.json")
    save_json(objects, "./data/unique_objects.json")
    print(' No. of unique objects:{}, \n No. of unique attributes:{}'.format(len(objects), len(attributes)))

    print(objects)
    print(attributes)

def get_vcpts_BOW():
    def BOW(concepts_set, vcpt_keys):
        bow = {key.strip():0 for key in vcpt_keys}
        unigrams = []
        for pair in concepts_set:
            items = pair.split()

            ##move to another function
            if len(items) == 3:
               unigrams.extend([items[0], "_".join([items[1], items[2]])])
            if len(items) == 2:
                unigrams.extend(items)
            if len(items) == 4:
                unigrams.extend(["_".join([items[0], items[1]]), "_".join([items[2], items[3]])])
            else:
                unigrams.extend(items)
            unigrams = [unigram.strip() for unigram in unigrams]

        for unigram in unigrams:
            if bow.has_key(unigram):
                bow[unigram] += 1
            else:
                bow[unigram] = 1
        return bow

    vcpts = load_pickle('./data/det_visual_concepts_hq.pickle')

    objects = load_json("./data/unique_objects.json")
    attr = load_json("./data/unique_attr.json")
    unique_concepts = objects + attr

    vcpt_dict = {key:None for key in vcpts.keys()}

    for k, v in tqdm(vcpts.items()):
        unique_pairs = set()
        for item in v:
            unique_pairs.update(item.lower().split(","))
        vcpt_dict[k] = BOW(unique_pairs, unique_concepts)

    save_json(vcpt_dict, "./data/vcpt_bow.json")

def get_qmask(q, q_l, tokenizer="custom"):
    qmask_part1 = np.zeros(q_l, dtype=int)

    split_index = get_split_index(list(q), tokenizer)
    if split_index is not None and split_index is not 0:
        qmask_part1[:split_index] = 1
        qmask_part2 = 1 - qmask_part1
    else:
        #approximate mask by splitting q in half
        qmask_part1[:q_l//2] = 1
        qmask_part2 = 1 - qmask_part1
    return qmask_part1, qmask_part2, split_index

def flip_question(q, q_l, tokenizer="custom"):
    flipped_q_final = np.zeros(q_l, dtype=int)
    qmask_part1 = np.zeros(q_l, dtype=int)
    if tokenizer is "custom":
        qmark_exists = 23509 in q
    elif tokenizer is "bert":
        qmark_exists = 1029 in q
    if qmark_exists:
        q_new, eos = q[:-2], q[-2:]
        q_l_new = q_l - 2
    else:
        q_new, eos = q[:-1], q[-1]
        q_l_new = q_l - 1

    # print(q, eos)
    flipped_q = np.zeros((q_l_new), dtype=int)

    split_index = get_split_index(q_new, tokenizer)

    if split_index is not None and split_index is not 0:
        qmask_part1[:q_l_new - split_index] = 1
        qmask_part2 = 1 - qmask_part1
        flipped_q[:q_l_new - split_index] = q_new[split_index:q_l_new]
        flipped_q[q_l_new - split_index:q_l_new] = q_new[:split_index]
        if qmark_exists:
            flipped_q_final[:-2] = flipped_q
            flipped_q_final[-2:] = eos
        else:
            flipped_q_final[:-1] = flipped_q
            flipped_q_final[-1] = eos
        assert(len(flipped_q_final) == len(qmask_part1))
        return flipped_q_final, qmask_part1, qmask_part2
    else:
        return q, None, None


    ######## uncomment to flip question ###########
    # for i in range(len(model_inputs[0])):
    #     q, q_l = model_inputs[0][i], model_inputs[1][i]
    #     flipped_q = torch.zeros(1, model_inputs[0][i].size(0)).long()
    #     split_index = get_split_index(q.tolist())
    #     # print(split_index)
    #     if split_index is not None:
    #         flipped_q[0,:q_l - split_index] = q[split_index:q_l]
    #         flipped_q[0, q_l - split_index:q_l] = q[:split_index]
    #         # print(token_to_sentence(flipped_q[0].tolist(), dset.idx2word))
    #         model_inputs[0][i] = flipped_q.to(opt.device)
    #     else:
    #         count += 1

def get_split_index(list_to_find, tokenizer="custom"):
    if tokenizer is "custom":
        if 7595 in list_to_find:
            return list_to_find.index(7595)
        elif 21437 in list_to_find:
            return list_to_find.index(21437)
        elif 4770 in list_to_find:
            return list_to_find.index(4770)
        elif 11923 in list_to_find:
            return list_to_find.index(11923)
    elif tokenizer is "bert":
        if 2043 in list_to_find:
            return list_to_find.index(2043) #when
        elif 2077 in list_to_find:
            return list_to_find.index(2077) #before
        elif 2044 in list_to_find:
            return list_to_find.index(2044) #after
        else:
            return -1

def get_verbs_and_nouns(text):
    import nltk
    from nltk import word_tokenize
    from nltk.tag import pos_tag

    tagged_sent = pos_tag(text.split())
    nouns, verbs = [], []
    _ = [nouns.append(word) for word, pos in tagged_sent if pos.startswith("NN") and word not in nouns]
    _ = [verbs.append(word) for word, pos in tagged_sent if pos.startswith("VB") and word not in verbs]


    return verbs, nouns

def get_all_img_ids(interval_start_img_id, interval_end_img_id, num_imgs, frame_interval=6):
    """ get 0.5fps image ids sequence that contains the localized img_ids
    this should be used for each question in bbt (since I made a stupid mistake T_T), note img_ids are 1-indexed
    :param interval_start_img_id: (int) the first img id used
    :param interval_end_img_id: (int) the last img id used
    :param num_imgs: (int) total number of images for the video
    :param frame_interval: (int)
    :return: indices (list), located_mask (list)
    """
    real_start = interval_start_img_id % frame_interval  # residual
    real_start = frame_interval if real_start == 0 else real_start
    indices = range(real_start, min(num_imgs+1, 301), frame_interval)
    assert 0 not in indices
    mask_start_idx = indices.index(interval_start_img_id)
    # mask_end_idx = indices.index(interval_end_img_id)
    # some indices are larger than num_imgs, TODO should be addressed in data preprocessing part
    if interval_end_img_id in indices:
        mask_end_idx = indices.index(interval_end_img_id)
    else:
        mask_end_idx = len(indices) - 1
    return indices, mask_start_idx, mask_end_idx


def merge_tvqa_and_tvqa_plus(mode):
    tvqa = load_json('./data/tvqa_{}_processed.json'.format(mode))
    tvqa_plus = load_json('./data/tvqa_plus_annotations/tvqa_plus_{}.json'.format(mode))
    framecnt_dic = load_json('./data/frm_cnt_cache.json')
    qid2ts = {}
    for item in tvqa_plus:
        qid2ts[item['qid']] = item['ts']

    for item in tqdm(tvqa):
        if item['qid'] in qid2ts.keys():
            # print(item['located_frame'])
            item['located_frame'] = interval2frame(qid2ts[item['qid']], framecnt_dic[item['vid_name']])
            # print(item['located_frame'])

    save_json(tvqa, './data/tvqa_{}_improved_ts.json'.format(mode))

if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # # parser.add_argument("--data_dir", type=str, default="./data", help="data dir path")
    # # parser.add_argument("--frm_dir", type=str,
    # #                     help="video frame dir path, the program will use provided cache if it exists. "
    # #                          "Only used to get number of extracted frames for each video.")
    # parser.add_argument("--vcpt_path", type=str, default="./data/det_visual_concepts_hq.pickle",
    #                          help="visual concepts feature path")
    # parser.add_argument("--vscene_path", type=str, default="./data/det_visual_scene_attributes_hq.pickle",
    #                          help="visual scene concepts feature path")
    # args = parser.parse_args()
    # vcpt_path = args.vcpt_path
    # vscene_path = args.vscene_path
    #
    # data_dir = args.data_dir
    # sub_dir = os.path.join(data_dir, "tvqa_subtitles")
    # raw_qa_files = glob.glob(os.path.join(data_dir, "tvqa_qa_release", "*jsonl"))
    # sub_cache_path = os.path.join(data_dir, "srt_data_cache.json")
    # frm_cnt_cache_path = os.path.join(data_dir, "frm_cnt_cache.json")
    # srt_data = load_srt(sub_dir, sub_cache_path)
    # srt_data = tokenize_srt(srt_data)
    #
    # for i, qa_file in enumerate(raw_qa_files):
    #     print("-"*60)
    #     print("Processing %s" % qa_file)
    #     processed_qa_path = os.path.join(data_dir, os.path.split(qa_file)[1].replace(".jsonl", "_processed.json"))
    #     process_qa(qa_file, srt_data, args.frm_dir, frm_cnt_cache_path, processed_qa_path)

    # group_data('train')
    # group_data('val')
    # get_visual_ques_stats()

    ##### uncomment the following code to clean up scene dic and save it #####
    # scene_dic = load_pickle(vscene_path)
    # scene_dic = clean_scene_lbls(scene_dic)
    # save_pickle(scene_dic, "./data/det_visual_scene_attributes_clean_hq.pickle")
    ##########################################################################

    ##### uncomment the following code to filter scene dic and save it #####
    # scene_dic = load_pickle(vscene_path)
    # scene_dic = filter_scene_lbls(scene_dic)    #check implementation to find filter criteria
    ##########################################################################

    ##### uncomment following line to clean scene labels and merge with visual concepts ###########
    # merge_visual_and_scene_cpts(vcpt_path=vcpt_path, vscene_path=vscene_path)

    ##### plot answer words frequency #########################
    # plot_answer_words_freq()

    ##### get #objs and #attr
    #get_unique_object_list()

    ######create BOW for vcpts ########
    #get_vcpts_BOW()
    # bow = load_json("./data/vcpt_bow.json")
    # print([len(value) for value in bow.values()])

    merge_tvqa_and_tvqa_plus('train')
    merge_tvqa_and_tvqa_plus('val')
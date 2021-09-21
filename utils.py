import random

__author__ = "Jie Lei"

import os
import json
# import pickle
import pickle
import torch


def read_json_lines(file_path):
    with open(file_path, "rb") as f:
        lines = []
        for l in f.readlines():
            loaded_l = json.loads(l.strip("\n"))
            lines.append(loaded_l)
    return lines


def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f)


def save_json_pretty(data, file_path):
    """save formatted json, use this one for some json config files"""
    with open(file_path, "w") as f:
        f.write(json.dumps(data, indent=4, sort_keys=True))


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def save_pickle(data, data_path):
    with open(data_path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f, encoding='latin1')


def mkdirp(p):
    if not os.path.exists(p):
        os.makedirs(p)


def files_exist(filepath_list):
    """check whether all the files exist"""
    for ele in filepath_list:
        if not os.path.exists(ele):
            return False
    return True


def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

########## my functions ################
def get_encoded_labels(bsz, k, idx):
    # from lbl k, generate mask [-1., -1., -1., -1.,  k-1]

    """

    :param bsz: int for batch size
    :param k: int: #classes
    :param idx: 1d tensor with length bsz
    :return:
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoded_label = torch.zeros(bsz, k).to(device)
    encoded_label.scatter_(1, idx.unsqueeze(1), 1)
    encoded_label = encoded_label + -1 * (1 - encoded_label * (k-1))
    return encoded_label


def select_best_preds(bsz, k, out_probs1, out_probs2):
    # generate masks based on classifier's predictions and compute relative confidence score
    # using relative distance cost function
    """

    :param bsz: int
    :param k: int #classes
    :param out_probs1: (bsz, k)
    :param out_probs2: (bsz, k)
    :return:
    """
    cval, cidx = out_probs2.max(dim=1)
    cls_lbl_enc = get_encoded_labels(bsz, k, cidx)
    cls_confidence = (out_probs2 * cls_lbl_enc).sum(dim=1)

    opval, opidx = out_probs1.max(dim=1)
    op_enc_lbls = get_encoded_labels(bsz, k, opidx)
    opt_confidence = (out_probs1 * op_enc_lbls).sum(dim=1)

    mask = (1 - (cls_confidence > opt_confidence)).long()
    first_idx = torch.arange(bsz, dtype=torch.long)
    concat = torch.cat([out_probs2.unsqueeze(2), out_probs1.unsqueeze(2)], dim=2).permute(0, 2,
                                                                                          1)  # outputs (bsz, 2, k)
    final_preds = concat[first_idx, mask]
    return final_preds

def get_chunks(l, n, m):
    """Yield successive n-sized chunks from l with overlap m."""
    try:
        for i in range(0, len(l), n-m):
            yield l[i:i + n]
    except:
        print("l:{}, n:{}, m:{}".format(l, n, m))


def get_show_name(clip_name):
    clip_name_parts = clip_name.split("_")
    if len(clip_name_parts) == 4:
        # video clip is from bbt
        show = 'bbt'
    else:
        show = clip_name_parts[0]  # first part is show name
    return show


def vid_2_ts():
    vid_2_ts = {}

    vid_2_ts = get_vid_2_ts_for_split('train', vid_2_ts)
    vid_2_ts = get_vid_2_ts_for_split('val', vid_2_ts)

    with open('data/vid2ts_dic.pkl', 'wb') as f:
        pickle.dump(vid_2_ts, f)

    return vid_2_ts


def get_vid_2_ts_for_split(split, vid_2_ts):
    with open('data/{}.pkl'.format(split), 'rb') as f:
        data = pickle.load(f)
    for qid, clip_name, question, answer, answer_embeddings, family, ts in data:
        key = clip_name+"_"+str(ts[0])+"_"+str(ts[1])
        vid_2_ts[key] = ts

    return vid_2_ts


def get_random_chunk_from_ts(tstamp, clip_len, full_clip_length):
    start, end = tstamp[0], tstamp[1]
    tstamp_duration = tstamp[1] - tstamp[0] + 1
    if tstamp_duration < clip_len:
        # print('oversample frames with clip_len')
        shortfall = clip_len - tstamp_duration
        new_start = max(0, start - shortfall//2)  # if clip_len = 32, ts = 10 frames, exceed = 22, oversample 11 frames
        # before actual ts i.e. -11+10 : 10 : 10 + 11
        new_end = min(new_start+clip_len, full_clip_length)
        new_start = max(0, new_end - clip_len)   # re-adjust new start
    elif tstamp_duration > clip_len:
        # print("randomly take consecutive frames of clip_len")
        exceed = tstamp_duration - clip_len
        new_start = start + random.randint(0, exceed-1)
        new_end = min(new_start+clip_len, full_clip_length)
        new_start = max(0, new_end - clip_len)  # re-adjust new start
    else:  # tstamp_duration == clip_len
        # print("equal length..")
        new_end = min(end, full_clip_length)
        new_start = new_end - clip_len + 1
        new_end = new_start + clip_len
    if new_end > full_clip_length:
        print('{}, {}, {}'.format(new_end, full_clip_length, tstamp))
    chunk = range(new_start, new_end)  # upperbound is exclusive in range object
    # print(chunk)
    return  chunk

####### borrowed code from allennlp ###############
##### https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py
def masked_softmax(vector,
                   mask,
                   dim= -1,
                   memory_efficient= False,
                   mask_fill_value= -1e32):
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


def get_proper_nouns(text):
    import nltk
    from nltk import word_tokenize
    from nltk.tag import pos_tag

    tagged_sent = pos_tag(text.split())
    properNouns = [word for word, pos in tagged_sent if pos == "NNP"]
    return properNouns

# Python code to sort the tuples using second element
# of sublist Function to sort using sorted()
def Sort(sub_li):

    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of
    # sublist lambda has been used
    return(sorted(sub_li, key = lambda x: x[1]))

##########################################################
require_clip = [
    'lstm_raw.rnn.weight_ih_l0',
    'lstm_raw.rnn.weight_hh_l0',
    'lstm_raw.rnn.bias_ih_l0',
    'lstm_raw.rnn.bias_hh_l0',
    'lstm_raw.rnn.weight_ih_l0_reverse',
    'lstm_raw.rnn.weight_hh_l0_reverse',
    'lstm_raw.rnn.bias_ih_l0_reverse',
    'lstm_raw.rnn.bias_hh_l0_reverse'
]


def clip_gradients(model, clip_value):
    for name, param in model.named_parameters():
        if name in require_clip:
            torch.nn.utils.clip_grad_norm_(param, clip_value)


def get_chunks_from_tstamps(frame_list, ts, clip_len=16, clip_overlap=0.5):

    #should not require frame_list but for now, let's keep it this way.
    frame_list = sorted(frame_list)
    # clip_len = 16
    # clip_overlap = 0.5
    overlap = int(clip_len * clip_overlap)

    all_chunks_for_clip = []

    # tstamp = ts[1]
    start, end = ts[0], ts[1]
    tstamp_duration = end - start
    exceed = clip_len - tstamp_duration

    updated_frame_list = frame_list[start:min(end, len(frame_list))]

    chunks_ = list(get_chunks(updated_frame_list, clip_len, overlap))

    for i in range(len(chunks_)):
        chunk = chunks_[i]

        if (len(chunk) < clip_len) and len(chunk)>1:
            new_start = max(chunk[0] - (clip_len - len(chunk)), 0)
            chunk = list(range(new_start, new_start+clip_len))
            chunks_[i] = chunk
    #
    # #remove chunks duplicates
    chunks_ = [ele for ind, ele in enumerate(chunks_) if ele not in chunks_[:ind]]

    return chunks_



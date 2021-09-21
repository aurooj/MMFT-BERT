import json

__author__ = "Jie Lei"
import io
import os
import h5py
import random
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
from torchvision.models.resnet import ResNet
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from utils import load_pickle, save_pickle, load_json, files_exist, get_show_name, get_random_chunk_from_ts
from preprocessing import flip_question, get_qmask
from transformers import BertTokenizer
from PIL import Image

def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)

    return x

class TVQADataset(Dataset):
    def __init__(self, mode="train"):
        self.raw_train = load_json('/data/Aisha/TVQA/Reason-VQA/data/tvqa_train_processed.json')
        # self.raw_train = self.raw_train[:250]
        self.raw_test = load_json('/data/Aisha/TVQA/Reason-VQA/data/tvqa_test_public_processed.json')
        self.raw_valid = load_json('/data/Aisha/TVQA/Reason-VQA/data/tvqa_val_processed.json')
        # self.raw_valid = self.raw_valid[:10]

        self.res101 = models.resnet101(pretrained=True)
        self.res101.eval()
        self.res101.forward = forward.__get__(self.res101, ResNet)
        self.transform = transforms.Compose([
                            transforms.Resize([224, 224]),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])])
        # attr_obj_pairs = self.get_obj_attr_pairs(self.vcpt_dict)
        self.vid_path = "../../data/tvqa_video_frames_fps3/"

        self.with_ts = True

        self.mode = mode
        self.cur_data_dict = self.get_cur_dict()


    def set_mode(self, mode):
        self.mode = mode
        self.cur_data_dict = self.get_cur_dict()


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
        items = {}
        # if self.with_ts: #changed by me to get tstamps anyway
        cur_start, cur_end = self.cur_data_dict[index]['located_frame']
        cur_vid_name = self.cur_data_dict[index]["vid_name"]

        if self.with_ts:
            show = get_show_name(cur_vid_name)
            path = os.path.join(self.vid_path, '{}_frames'.format(show), cur_vid_name)
            frame_list = os.listdir(path)[cur_start:cur_end+1]
            if len(frame_list) == 0:
                frame_list = os.listdir(path)[:10] #take first 10 frames

            vid_frames = []

            for i in range(0, len(frame_list), 3):#1 fps
                image = Image.open(os.path.join(path, frame_list[i])).convert('RGB')
                image = self.transform(image).unsqueeze(0)
                vid_frames.append(image)

        items['vid_name'] = cur_vid_name + "_" + str(self.cur_data_dict[index]['qid'])
        items['features'] = vid_frames
        return items

batch_size = 4

def create_dataset(split):
    dataloader = DataLoader(TVQADataset(split), batch_size=batch_size,
                            num_workers=4)

    size = len(dataloader)

    print(split, 'total', size * batch_size)

    f = h5py.File('../../data/{}_res101_pooled_features.hdf5'.format(split), 'w', libver='latest')
    dset = f.create_dataset('data', (size * batch_size, 1024, 14, 14),
                            dtype='f4')
    vid2idx = {}
    with torch.no_grad():
        for i, item in tqdm(enumerate(dataloader)):
            cnt = i * batch_size
            for j in range(min(batch_size, len(item['vid_name']))):
                vid2idx[item['vid_name'][j]] = cnt
                cnt = cnt+1

            if np.shape(item['features'])[0] != batch_size:# if last batch when total data%bsz !=0
                end = i+1 +  np.shape(item['features'])[0]
                print(end)
            else:
                end = (i + 1) * batch_size

            vid_frames_batch = item['features']
            vid_frame_array = torch.cat(vid_frames, dim=1)
            vid_features = res101(vid_frame_array.view(-1, 3, 224, 224)).detach().cpu().numpy()
            vid_features = vid_features.view(batch_size, -1, 1024, 14, 14)

            pooled_feature = np.max(vid_features, axis=0)
            dset[i * batch_size:end] = item['features']

    f.close()
    save_pickle(vid2idx, '../../data/{}_res101_vid2idx.pkl'.format(split))

# create_dataset('valid')
create_dataset('train')


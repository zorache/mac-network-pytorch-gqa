import json
import os
import pickle

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import h5py

from transforms import Scale

img = None
img_info = {}
def gqa_feature_loader(config):
    global img, img_info
    if img is not None:
        return img, img_info
    if config.objects and not config.spatial:
        h = h5py.File('/scratch3/zche/GQA/processed/objects.h5', 'r')
        img = h['features']
        img_info = json.load(open('/scratch3/zche/GQA/processed/objects_merged_info.json', 'r'))
    elif config.spatial and not config.objects:
        h = h5py.File('/scratch3/zche/GQA/processed/spatial.h5', 'r')
        img = h['features']
        img_info = json.load(open('/scratch3/zche/GQA/processed/spatial_merged_info.json', 'r'))

    return img, img_info

def bert_feature_loader(config, split):
    global lengths, outputs, state
    if config.bert:
        h = h5py.File(f'/scratch3/zche/GQA/processed/bert_features_{split}.h5', 'r')
        lengths = h['lengths']
        outputs = h['outputs']
        state = h['state']
        return lengths, outputs, state
    else:
        return None, None, None


class GQA(Dataset):
    def __init__(self, root, config,split='train', transform=None):
        with open(f'/scratch3/zche/GQA/processed/gqa_{split}.pkl', 'rb') as f:
            self.data = pickle.load(f)
        self.config=config
        self.root = root
        self.split = split
        self.img, self.img_info = gqa_feature_loader(config)
        self.bertlength, self.bertoutputs, self.bertstate = bert_feature_loader(self.config, split)

    def __getitem__(self, index):
        imgfile, question, answer = self.data[index]
        idx = int(self.img_info[imgfile]['index'])
        img = torch.from_numpy(self.img[idx])
        if self.config.bert:
            bertlength = self.bertlength[index]
            bertoutputs = torch.from_numpy(self.bertoutputs[index])
            bertstate = torch.from_numpy(self.bertstate[index])
        else:
            return img, question, len(question)
        return img, question, len(question), answer, bertlength, bertoutputs, bertstate

    def __len__(self):
        return len(self.data)

transform = transforms.Compose([
    Scale([224, 224]),
    transforms.Pad(4),
    transforms.RandomCrop([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
])

def collate_data(batch):
    # Create padding for data for GPU operations
    images, lengths, answers = [], [], []
    batch_size = len(batch)

    max_len = max(map(lambda x: len(x[1]), batch))

    questions = np.zeros((batch_size, max_len), dtype=np.int64)
    
    sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

    b_lengths = np.zeros((batch_size, 1), dtype=np.int64) 
    b_states = np.zeros((batch_size, 768), dtype=np.float32) 
    b_outputs = np.zeros((batch_size, 30, 768), dtype=np.float32) 
    num=len(batch[0])   
    if num>5:
        for i, b in enumerate(sort_by_len):
            image, question, length, answer, b_length, b_output, b_state = b
            images.append(image)
            length = len(question)
            questions[i, :length] = question
            lengths.append(length)
            answers.append(answer)
            b_lengths[i]=b_length
            b_states[i]=b_state
            b_outputs[i]=b_output
        return torch.stack(images), torch.from_numpy(questions), \
                lengths, torch.LongTensor(answers), \
                torch.from_numpy(b_lengths), \
                torch.from_numpy(b_outputs),\
                torch.from_numpy(b_states)
    else:
        for i, b in enumerate(sort_by_len):
            #image, question, length, answer = b
            image, question, answer = b
            images.append(image)
            length = len(question)
            questions[i, :length] = question
            lengths.append(length)
            answers.append(answer)
        return torch.stack(images), torch.from_numpy(questions), \
                lengths, torch.LongTensor(answers)

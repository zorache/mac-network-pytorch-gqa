import os
import sys
import json
import pickle
import h5py

from typing import List, Any
import numpy as np
import torch
from torch import LongTensor
import nltk
import tqdm
from torchvision import transforms
from PIL import Image
from transforms import Scale
from transformers import BertTokenizer, BertModel
image_index = {#'CLEVR': 'image_filename',
               'gqa': 'imageId'}

# Preprocess json training and validation questions by tokenizing the questions and saving
# image, tokenized questions, and answers in a list

# python preprocess.py gqa <path with questions folder with train/val/test> <type>
# example: python preprocess.py gqa /scratch3/zche/GQA/raw/ bert

#example: python preprocess.py gqa /scratch3/zche/GQA/raw/ bert_features

def process_question(root, split, word_dic=None, answer_dic=None, dataset_type='GQA'):
    if word_dic is None:
    	word_dic = {}
    if answer_dic is None:
        answer_dic = {}
    with open(os.path.join(root, 'questions', f'{split}_balanced_questions.json')) as f:
        data = json.load(f)
    result = []
    word_index = 1
    answer_index = 0
    n =0
    for question in tqdm.tqdm(data):
        #smaller subsets
        if split=='train' and n==200000:
            break
        if split=='val' and n==40000:
            break
        words = nltk.word_tokenize(data[question]['question']) #tokenize each word 
        question_token = []
        for word in words:
            try:
                question_token.append(word_dic[word])
            except:
                question_token.append(word_index)
                word_dic[word] = word_index
                word_index += 1
        answer_word = data[question]['answer']
        try:
            answer = answer_dic[answer_word]
        except:
            answer = answer_index
            answer_dic[answer_word] = answer_index
            answer_index += 1
        result.append((data[question][image_index[dataset_type]], question_token, answer))
        n=n+1
    with open(f'/scratch3/zche/GQA/processed/{dataset_type}_{split}_1.pkl', 'wb') as f:
        pickle.dump(result, f)
    return word_dic, answer_dic




def process_question_spatial(root, split, word_dic=None, answer_dic=None, dataset_type='GQA'):
    if word_dic is None:
    	word_dic = {}
    if answer_dic is None:
        answer_dic = {}
    with open(os.path.join(root, 'questions', f'{split}_balanced_questions.json')) as f:
        data = json.load(f)
    result = []
    word_index = 1
    answer_index = 0
    n =0
    for question in tqdm.tqdm(data):
        #smaller subsets
        if split=='train' and n==200000:
            break
        if split=='val' and n==40000:
            break
        words = nltk.word_tokenize(data[question]['question']) #tokenize each word 
        question_token = []
        for word in words:
            try:
                question_token.append(word_dic[word])
            except:
                question_token.append(word_index)
                word_dic[word] = word_index
                word_index += 1
        answer_word = data[question]['answer']
        try:
            answer = answer_dic[answer_word]
        except:
            answer = answer_index
            answer_dic[answer_word] = answer_index
            answer_index += 1
        result.append((data[question][image_index[dataset_type]], question_token, answer))
        n=n+1
    with open(f'/scratch3/zche/GQA/processed/{dataset_type}_{split}_1.pkl', 'wb') as f:
        pickle.dump(result, f)
    return word_dic, answer_dic

def process_question_bert(root, split, word_dic=None, answer_dic=None, dataset_type='GQA'):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if word_dic is None:
    	word_dic = {}
    if answer_dic is None:
        answer_dic = {}
    with open(os.path.join(root, 'questions', f'{split}_balanced_questions.json')) as f:
        data = json.load(f)
    result = []
    word_index = 1
    answer_index = 0
    n =0
    for question in tqdm.tqdm(data):#['questions']):
        #smaller subsets
        if split=='train' and n==200000:
            break
        if split=='val' and n==40000:
            break
        words = tokenizer.encode(data[question]['question'])
        question_token = []
        for word in words:
            try:
                question_token.append(word_dic[word])
            except:
                question_token.append(word_index)
                word_dic[word] = word_index
                word_index += 1
        answer_word = data[question]['answer']
        try:
            answer = answer_dic[answer_word]
        except:
            answer = answer_index
            answer_dic[answer_word] = answer_index
            answer_index += 1
        result.append((data[question][image_index[dataset_type]], question_token, answer))
        n=n+1
    with open(f'/scratch3/zche/GQA/processed/{dataset_type}_{split}_bert_token.pkl', 'wb') as f:
        pickle.dump(result, f)
    return word_dic, answer_dic


#Saving a h5py file of bert features for training/val set
def process_question_bert_features(root, split, word_dic=None, answer_dic=None, dataset_type='GQA'):
    device = torch.device("cuda")
    with open(os.path.join(root, 'questions', f'{split}_balanced_questions.json')) as f:
        data = json.load(f)
    if split=='train':
        subset =200000
    elif split=='val':
        subset = 40000
    elif split=='test':
        subset = 40000
    with h5py.File(os.path.join('/scratch3/zche/GQA/processed', f"gqa_{split}_bert_features.h5"), 'w') as f:
        outputs = f.create_dataset("outputs", (subset, 30, 768), dtype=np.float32)
        lengths = f.create_dataset("lengths", (subset, ), dtype=np.int)
        states = f.create_dataset("state", (subset, 768), dtype=np.float32)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #BertModel().to(device)
        model = BertModel.from_pretrained('bert-base-uncased').to(device)
        model.to(device)
        print(model.device)
        #model.cuda()
        batch_size = 64
        keys = list(data.keys())
        for start in tqdm.tqdm(range(0, subset, batch_size), desc="Extracting BERT features"):
            end = min(start + batch_size, subset)
            input_ids = [tokenizer.encode(data[keys[i]]['question'], max_length=30) for i in range(start, end)]
            padded_input_ids, _ = pad_sequence(input_ids, padding_value=tokenizer.pad_token_id, max_len=30, output_tensor=True)
            #torch.tensor(padded_input_ids).cuda()
            padded_input_ids.to(device)
            padded_input_ids.cuda()
            out = model(padded_input_ids)
            lengths[start:end] = np.array([len(s) for s in input_ids])
            outputs[start:end] = out[0].detach().cpu().numpy()
            states[start:end] = out[1].detach().cpu().numpy()
    

def pad_sequence(
        data: List[List[Any]],
        padding_value,
        max_len: int = None,
        output_tensor: bool = False,
        dim: int = 2):

    max_len = max_len or max([len(seq) for seq in data])

    i = 0
    while len(data[i]) == 0:
        i += 1
        if i == len(data):
            raise ValueError("Empty input.")
    if isinstance(data[i][0], list) or isinstance(data[i][0], tuple):
        padding_value = [padding_value for _ in range(len(data[i][0]))]

    if not output_tensor:
        lengths = [max(len(seq), 1) for seq in data]
        if type(data[0]) == list:
            data = [torch.tensor(seq + [padding_value] * (max_len - len(seq))) for seq in data]
        else:
            data = [torch.cat([seq, torch.tensor([padding_value] * (max_len - len(seq)))]) for seq in data]
        return data, lengths
    else:
        lengths = [max(len(seq), 1) for seq in data]
        data = [seq + [padding_value] * (max_len - len(seq)) for seq in data]
        return torch.tensor(data), LongTensor(lengths)



if __name__ == '__main__':
    dataset_type = sys.argv[1]
    root = sys.argv[2]
    choice = sys.argv[3]   #Specify which type of preprocessing [token, bert, bert_features]
    if choice=='bert':
        print('Preprocessing with BERT token')
        word_dic, answer_dic = process_question_bert(root, 'train', dataset_type=dataset_type)
        process_question_bert(root, 'val', word_dic, answer_dic, dataset_type=dataset_type)
    elif choice=='token':
        word_dic, answer_dic = process_question(root, 'train', dataset_type=dataset_type)
        process_question(root, 'val', word_dic, answer_dic, dataset_type=dataset_type)
    elif choice=='bert_features':
        process_question_bert_features(root, 'train', dataset_type=dataset_type)
        process_question_bert_features(root, 'val', word_dic, answer_dic, dataset_type=dataset_type)
    with open(f'/scratch3/zche/GQA/processed/{dataset_type}_dic.pkl', 'wb') as f:
        pickle.dump({'word_dic': word_dic, 'answer_dic': answer_dic}, f)

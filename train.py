import multiprocessing
import pickle
import sys

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import collate_data, transform, GQA
from model_gqa import MACNetwork

import h5py
import os

import wandb
wandb.init(project="test-drive", config={
    "learning_rate": 0.0001,
    "encInputDropout" : 0.2, # dropout of the rnn inputs to the Question Input Unit
    "encStateDropout" : 0.0, # dropout of the rnn states of the Question Input Unit
    "stemDropout" : 0.2, # dropout of the Image Input Unit (the stem)
    "qDropout" : 0.08, # dropout on the question vector
    "qDropoutOut" : 0, # dropout on the question vector the goes to the output unit
    "memoryDropout" : 0.15, # dropout on the recurrent memory
    "readDropout" : 0.15, # dropout of the read unit
    "writeDropout" : 1.0, # dropout of the write unit
    "outputDropout" : 0.85, # dropout of the output unit
    "controlPreDropout" : 1.0, # dropout of the write unit
    "controlPostDropout" : 1.0, # dropout of the write unit
    "wordEmbDropout" : 1.0, # dropout of the write unit
    "subset_train": 200000,
    "objects":False,
    "spatial":True,
    "epochs":10,
    "max_step":5,
    "architecture": "MAC",
    "version": 'spatial1',
    "bert": False,
    "encoderdim": 768,
    "dataset": "GQA",
})

config = wandb.config

batch_size = 128
n_epoch = config.epochs
dim_dict = {'CLEVR': 512,
            'gqa': 2048}

device = torch.device('cuda:2') #if torch.cuda.is_available() else 'cpu')
print(device)


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def train(epoch, dataset_type, config):
    if dataset_type == "CLEVR":
        dataset_object = CLEVR('data/CLEVR_v1.0', transform=transform)
    else:
        dataset_object = GQA('data/gqa', config=config, transform=transform)

    train_set = DataLoader(
        dataset_object, batch_size=batch_size, num_workers=multiprocessing.cpu_count(), collate_fn=collate_data
    )

    dataset = iter(train_set)
    pbar = tqdm(dataset)
    moving_loss = 0

    net.train(True)
    if config.bert:
       for image, question, q_len, answer, b_length, b_outputs, b_state in pbar:
            image, question, answer, b_length, b_outputs, b_state = (
                image.to(device),
                question.to(device),
                answer.to(device),
                b_length.to(device),
                b_outputs.to(device),
                b_state.to(device),
            )

            net.zero_grad()
            output = net(image, question, q_len, b_length=b_length, b_outputs=b_outputs,b_state=b_state)
            loss = criterion(output, answer)
            loss.backward()
            optimizer.step()
            correct = output.detach().argmax(1) == answer
            correct = torch.tensor(correct, dtype=torch.float32).sum() / batch_size

            if moving_loss == 0:
                moving_loss = correct
            else:
                moving_loss = moving_loss * 0.99 + correct * 0.01

            pbar.set_description('Epoch: {}; Loss: {:.8f}; Acc: {:.5f}'.format(epoch + 1, loss.item(), moving_loss))
            wandb.log({"acc":correct, "loss":loss})
            accumulate(net_running, net)
    else:
        for image, question, q_len, answer in pbar:
            image, question, answer = (
                image.to(device),
                question.to(device),
                answer.to(device),
            )

            net.zero_grad()
            output = net(image, question, q_len)
            loss = criterion(output, answer)
            loss.backward()
            optimizer.step()
            correct = output.detach().argmax(1) == answer
            correct = torch.tensor(correct, dtype=torch.float32).sum() / batch_size

            if moving_loss == 0:
                moving_loss = correct
            else:
                moving_loss = moving_loss * 0.99 + correct * 0.01

            pbar.set_description('Epoch: {}; Loss: {:.8f}; Acc: {:.5f}'.format(epoch + 1, loss.item(), moving_loss))
            wandb.log({"acc":correct, "loss":loss})
            accumulate(net_running, net)
        
    if dataset_type=="CLEVR":
        dataset_object.close()


def valid(epoch, dataset_type, config):
    if dataset_type == "CLEVR":
        dataset_object = CLEVR('data/CLEVR_v1.0', 'val', transform=None)
    else:
        dataset_object = GQA('data/gqa', split='val', config=config,transform=None)

    valid_set = DataLoader(
        dataset_object, batch_size=4*batch_size, num_workers=multiprocessing.cpu_count(), collate_fn=collate_data
    )
    dataset = iter(valid_set)
    net_running.train(False)
    correct_counts = 0
    total_counts = 0
    running_loss = 0.0
    batches_done = 0
    with torch.no_grad():
        pbar = tqdm(dataset)
        if config.bert:
            for image, question, q_len, answer, b_length, b_outputs, b_state in pbar:
                image, question, answer, b_length, b_outputs, b_state = (
                    image.to(device),
                    question.to(device),
                    answer.to(device),
                    b_length.to(device),
                    b_outputs.to(device),
                    b_state.to(device),
                )
                output = net(image, question, q_len, b_length=b_length, b_outputs=b_outputs,b_state=b_state)
                loss = criterion(output, answer)
                correct = output.detach().argmax(1) == answer
                running_loss += loss.item()

                batches_done += 1
                for c in correct:
                    if c:
                        correct_counts += 1
                    total_counts += 1

                pbar.set_description('Epoch: {}; Loss: {:.8f}; Acc: {:.5f}'.format(epoch + 1, loss.item(), correct_counts / batches_done))
        else:
            for image, question, q_len, answer in pbar:
                image, question, answer = (
                    image.to(device),
                    question.to(device),
                    answer.to(device),
                )
                output = net_running(image, question, q_len)
                loss = criterion(output, answer)
                correct = output.detach().argmax(1) == answer
                running_loss += loss.item()

                batches_done += 1
                for c in correct:
                    if c:
                        correct_counts += 1
                    total_counts += 1

                pbar.set_description('Epoch: {}; Loss: {:.8f}; Acc: {:.5f}'.format(epoch + 1, loss.item(), correct_counts / batches_done))
    
    folder_name = f'log/{config.subset_train}_{config.epochs}_{config.version}'
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    with open(folder_name+'/log_{}.txt'.format(str(epoch + 1).zfill(2)), 'w') as w:
        w.write('{:.5f}\n'.format(correct_counts / total_counts))

    print('Validation Accuracy: {:.5f}'.format(correct_counts / total_counts))
    print('Validation Loss: {:.8f}'.format(running_loss / total_counts))

    if dataset_type=="CLEVR":
        dataset_object.close()


if __name__ == '__main__':
    dataset_type = sys.argv[1]
    with open(f'/scratch3/zche/GQA/processed/{dataset_type}_dic.pkl', 'rb') as f:
        dic = pickle.load(f)

    n_words = len(dic['word_dic']) + 1
    n_answers = len(dic['answer_dic'])

    net = MACNetwork(n_words, dim_dict[dataset_type], classes=n_answers, max_step=wandb.config.max_step,config=wandb.config).to(device)
    net_running = MACNetwork(n_words, dim_dict[dataset_type], classes=n_answers, max_step=wandb.config.max_step,config=wandb.config).to(device)
    accumulate(net_running, net, 0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)

    for epoch in range(n_epoch):
        train(epoch, dataset_type, config)
        valid(epoch, dataset_type, wandb.config)
        folder_name = f'checkpoint/{config.subset_train}_{config.epochs}_{config.version}'
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        with open(folder_name+'/checkpoint_{}.model'.format(str(epoch + 1).zfill(2)), 'wb') as f:
            torch.save(net_running.state_dict(), f)
    wandb.finish()
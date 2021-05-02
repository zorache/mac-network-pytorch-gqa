import torch
import torch.nn.functional as F
#from block.models.networks.fusions.fusions import Tucker
from torch import nn
from torch.nn.init import kaiming_uniform_, xavier_uniform_

import config
from utils import get_or_load_embeddings


from collections import namedtuple


batch_x_fields = (
    "images",
    "question_ids",
    "questions",
    "question_lengths",
    "question_bert_outputs",
    "question_bert_states",
    "question_bert_lengths",
    "objects",
    "object_lengths",
    "object_bounding_boxes",
    "object_identities",
    "object_attributes",
    "object_relations")

#Batch_Bert = namedtuple("Batch_Bert", batch_x_fields) defaults=(None,) * len(batch_x_fields))



def linear(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()

    return lin

class ControlUnit(nn.Module):
    def __init__(self, dim, max_step):
        super().__init__()

        self.position_aware = nn.ModuleList()
        for i in range(max_step):
            self.position_aware.append(linear(dim * 2, dim))

        self.control_question = linear(dim * 2, dim)
        self.attn = linear(dim, 1)

        self.dim = dim

    def forward(self, step, context, question, control):
        #print(control.shape)
        #print(control.shape)
        #print(control.shape)
        #print(control.shape)
        #print(control.shape)
        #print(control.shape)
        position_aware = self.position_aware[step](question)
        #print(control.shape)
        #print(control.shape)
        #print(control.shape)
        #print(control.shape)
        #print(control.shape)
        #print(control.shape)
        control_question = torch.cat([control, position_aware], 1)
        control_question = self.control_question(control_question)
        control_question = control_question.unsqueeze(1)

        context_prod = control_question * context
        attn_weight = self.attn(context_prod)

        attn = F.softmax(attn_weight, 1)

        next_control = (attn * context).sum(1)

        return next_control


# class ReadUnit(nn.Module):
#     def __init__(self, dim):
#         super().__init__()

#         self.mem = linear(dim, dim)
#         self.concat = linear(dim * 2, dim)
#         self.attn = linear(dim, 1)
#         self.tucker = Tucker((2048, 2048), 1, mm_dim=50, shared=True)

#     def forward(self, memory, know, control):
#         mem = self.mem(memory[-1]).unsqueeze(2)
#         s_matrix = (mem * know)
#         s_matrix = s_matrix.view(-1, 2048)
#         attn = self.tucker([s_matrix, control[-1].repeat(know.size(2), 1)]).view(know.size(2), know.size(0))
#         attn = attn.transpose(0, 1)
#         attn = F.softmax(attn, 1).unsqueeze(1)
#         read = (attn * know).sum(2)

#         return read


class ReadUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.mem = linear(dim, dim)
        self.concat = linear(dim * 2, dim)
        self.attn = linear(dim, 1)

    def forward(self, memories, k, c):
        """
        :param memories:
        :param k: knowledge
        :param c: control
        :return: r_i
        """
        # 1. Interaction between knowledge k_{h,w} and memory m_{i-1}
        m_prev = memories[-1]
        I = self.mem(m_prev).unsqueeze(2) * k  # I_{i,h,w}

        # 2. Calculate I'_{i,h,w}
        I = self.concat(torch.cat([I, k], 1).permute(0, 2, 1))

        # 3. Attention distribution over knowledge base elements
        attn = I * c[-1].unsqueeze(1)
        attn = self.attn(attn).squeeze(2)
        attn = F.softmax(attn, 1).unsqueeze(1)
        read = (attn * k).sum(2)
        return read

class WriteUnit(nn.Module):
    def __init__(self, dim, self_attention=False, memory_gate=False):
        super().__init__()

        self.concat = linear(dim * 2, dim)

        if self_attention:
            self.attn = linear(dim, 1)
            self.mem = linear(dim, dim)

        if memory_gate:
            self.control = linear(dim, 1)

        self.self_attention = self_attention
        self.memory_gate = memory_gate

    def forward(self, memories, retrieved, controls):
        prev_mem = memories[-1]
        concat = self.concat(torch.cat([retrieved, prev_mem], 1))
        next_mem = concat

        if self.self_attention:
            controls_cat = torch.stack(controls[:-1], 2)
            attn = controls[-1].unsqueeze(2) * controls_cat
            attn = self.attn(attn.permute(0, 2, 1))
            attn = F.softmax(attn, 1).permute(0, 2, 1)

            memories_cat = torch.stack(memories, 2)
            attn_mem = (attn * memories_cat).sum(2)
            next_mem = self.mem(attn_mem) + concat

        if self.memory_gate:
            control = self.control(controls[-1])
            gate = F.sigmoid(control)
            next_mem = gate * prev_mem + (1 - gate) * next_mem

        return next_mem


class MACUnit(nn.Module):
    def __init__(self, dim, max_step=12,
                self_attention=False, memory_gate=False,
                dropout=0.15):
        super().__init__()

        self.control = ControlUnit(dim, max_step)
        self.read = ReadUnit(dim)
        self.write = WriteUnit(dim, self_attention, memory_gate)

        self.mem_0 = nn.Parameter(torch.zeros(1, dim))
        self.control_0 = nn.Parameter(torch.zeros(1, dim))

        self.dim = dim
        self.max_step = max_step
        self.dropout = dropout
        # self.dropouts = {}
        # self.dropouts["encInput"]: config.encInputDropout
        # self.dropouts["encState"]: config.encStateDropout
        # self.dropouts["stem"]: config.stemDropout
        # self.dropouts["question"]: config.qDropout
        # self.dropouts["memory"]: config.memoryDropout
        # self.dropouts["read"]: config.readDropout
        # self.dropouts["write"]: config.writeDropout
        # self.dropouts["output"]: config.outputDropout
        # self.dropouts["controlPre"]: config.controlPreDropout
        # self.dropouts["controlPost"]: config.controlPostDropout
        # self.dropouts["wordEmb"]: config.wordEmbDropout
        # self.dropouts["word"]: config.wordDp
        # self.dropouts["vocab"]: config.vocabDp
        # self.dropouts["object"]: config.objectDp
        # self.dropouts["wordStandard"]: config.wordStandardDp

    def get_mask(self, x, dropout):
        mask = torch.empty_like(x).bernoulli_(1 - dropout)
        mask = mask / (1 - dropout)

        return mask

    def forward(self, context, question, knowledge):
        b_size = question.size(0)

        control = self.control_0.expand(b_size, self.dim)
        memory = self.mem_0.expand(b_size, self.dim)

        if self.training:
            control_mask = self.get_mask(control, self.dropout)
            memory_mask = self.get_mask(memory, self.dropout)
            control = control * control_mask
            memory = memory * memory_mask

        controls = [control]
        memories = [memory]

        for i in range(self.max_step):
            control = self.control(i, context, question, control)
            if self.training:
                control = control * control_mask
            controls.append(control)

            read = self.read(memories, knowledge, controls)
            memory = self.write(memories, read, controls)
            if self.training:
                memory = memory * memory_mask
            memories.append(memory)

        return memory


class MACNetwork(nn.Module):
    def __init__(self, n_vocab, dim, config, embed_hidden=300,
                max_step=12, self_attention=False, memory_gate=False,
                classes=28, dropout=0.15):
        super().__init__()
        self.config=config
        self.conv = nn.Sequential(nn.Conv2d(1024, dim, 3, padding=1),
                                nn.ELU(),
                                nn.Conv2d(dim, dim, 3, padding=1),
                                nn.ELU())
        

        self.encoder_output_linear = nn.Linear(config.encoderdim, 2048)
        self.encoder_state_linear = nn.Linear(config.encoderdim, 4096)

        self.embed = nn.Embedding(n_vocab, embed_hidden)
        self.embed.weight.data = torch.Tensor(get_or_load_embeddings())
        self.embed.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_hidden, dim,
                        batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(dim * 2, dim)

        self.mac = MACUnit(dim, max_step,
                        self_attention, memory_gate, dropout)


        self.classifier = nn.Sequential(linear(dim * 3, dim),
                                        nn.ELU(),
                                        linear(dim, classes))

        self.max_step = max_step
        self.dim = dim

        self.reset()

    def reset(self):
        self.embed.weight.data.uniform_(0, 1)

        kaiming_uniform_(self.conv[0].weight)
        self.conv[0].bias.data.zero_()
        kaiming_uniform_(self.conv[2].weight)
        self.conv[2].bias.data.zero_()

        kaiming_uniform_(self.classifier[0].weight)

    def forward(self, image, question, question_len, b_length=None, b_outputs=None, b_state=None ,dropout=0.15):
        b_size = question.size(0)
        img = image
        img = img.view(b_size, self.dim, -1)

        if self.config.bert:
            b_state=b_state.cuda()
            b_outputs = b_outputs.cuda()
            #use BERT embeddings
            h = self.encoder_state_linear(b_state)
            print('here')
            c = self.encoder_output_linear(b_outputs) 
            print('passed')
        else:
            #embed with bidirectional LSTM
            embed = self.embed(question)
            embed = nn.utils.rnn.pack_padded_sequence(embed, question_len,
                                                    batch_first=True)
            lstm_out, (h, _) = self.lstm(embed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
            lstm_out = self.lstm_proj(lstm_out)
            c= lstm_out
            h = h.permute(1, 0, 2).contiguous().view(b_size, -1)   

        #MAC cell 
        #wandb.log({'h': h.shape, 'epoch': 5})
        #wandb.log({'c': c.shape, 'epoch': 5})
        print("h:"+str(h.shape))  #[128, 4096]
        print("c:"+str(c.shape))  #[128, x , 2048]
        memory = self.mac(c, h, img)
        
        #Output
        out = torch.cat([memory, h], 1)
        out = self.classifier(out)

        return out
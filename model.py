import torch
from torch.autograd import Variable
from torch import nn
from torch.nn.init import kaiming_uniform_, xavier_uniform_, normal
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        position_aware = self.position_aware[step](question)

        control_question = torch.cat([control, position_aware], 1)
        control_question = self.control_question(control_question)
        control_question = control_question.unsqueeze(1)

        context_prod = control_question * context
        attn_weight = self.attn(context_prod)

        attn = F.softmax(attn_weight, 1)

        next_control = (attn * context).sum(1)

        return next_control


class ReadUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.mem = linear(dim, dim)
        self.concat = linear(dim * 2, dim)
        self.attn = linear(dim, 1)

    def forward(self, memory, know, control):
        mem = self.mem(memory[-1]).unsqueeze(2)
        concat = self.concat(torch.cat([mem * know, know], 1) \
                                .permute(0, 2, 1))
        attn = concat * control[-1].unsqueeze(1)
        attn = self.attn(attn).squeeze(2)
        attn = F.softmax(attn, 1).unsqueeze(1)

        read = (attn * know).sum(2)

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
    def __init__(self, n_vocab, dim, opt, embed_hidden=300,
                 max_step=12, self_attention=False, memory_gate=False,
                 classes=5, dropout=0.15):
        super().__init__()
        print('Max steps:{}'.format(max_step))
        if opt.feat in ['resnet']:
            self.conv = nn.Sequential(nn.Conv2d(1024, dim, 3, padding=1),
                                    nn.ELU(),
                                    nn.Dropout2d(0.6),
                                    nn.Conv2d(dim, dim, 3, padding=1),
                                    nn.ELU(),
                                    nn.Dropout2d(0.6))
        if opt.feat in ['c3d','i3d']:
            self.conv = nn.Sequential(nn.Conv3d(1024, dim, 3, padding=1),
                                      nn.ELU(),
                                      nn.Dropout3d(0.5),
                                      nn.Conv3d(dim, dim, 3, padding=1),
                                      nn.ELU(),
                                      nn.Dropout3d(0.5),)

        self.embed = nn.Embedding(n_vocab, embed_hidden)
        self.embed.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_hidden, dim,
                        batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(dim * 2, dim)

        self.mac = MACUnit(dim, max_step,
                        self_attention, memory_gate, dropout)

        self.linear = nn.Sequential(linear(embed_hidden, dim),
                                    nn.ELU(),
                                    nn.Dropout(dropout),
                                    linear(dim, dim),
                                    nn.ELU(),
                                    nn.Dropout(dropout)
                                    )
        self.final_memory = nn.Sequential(linear(dim, dim),
                                    nn.ELU(),
                                    nn.Dropout(dropout),
                                    linear(dim, dim),
                                    nn.ELU(),
                                    nn.Dropout(dropout),
                                    )

        self.cos = nn.CosineSimilarity(eps=1e-6)

        self.classifier = nn.Sequential(linear(dim * 3+classes, dim),
                                        nn.ELU(),
                                        nn.Dropout(dropout),
                                        linear(dim, classes))

        self.max_step = max_step
        self.dim = dim
        self.opt = opt
        self.classes = classes

        self.reset()

    def reset(self):
        if not self.opt.no_glove:
            self.load_embedding(self.opt.vocab_embedding)
        else:
            self.embed.weight.data.uniform_(0, 1)

        kaiming_uniform_(self.conv[0].weight)
        self.conv[0].bias.data.zero_()
        kaiming_uniform_(self.conv[3].weight)
        self.conv[3].bias.data.zero_()

        kaiming_uniform_(self.classifier[0].weight)

    def load_embedding(self, pretrained_embedding):
        self.embed.weight.data.copy_(torch.from_numpy(pretrained_embedding))



    def forward(self, image, question, question_len, answer_embeddings, answer_len, dropout=0.15):
        b_size = question.size(0)
        image = F.normalize(image, p=2, dim=1) # normalize features before feeding to conv
        img = self.conv(image)
        # img = image
        img = img.view(b_size, self.dim, -1)

        embed = self.embed(question)
        embed = nn.utils.rnn.pack_padded_sequence(embed, question_len,
                                                batch_first=True)
        lstm_out, (h, _) = self.lstm(embed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out,
                                                    batch_first=True)
        lstm_out = self.lstm_proj(lstm_out)
        h = h.permute(1, 0, 2).contiguous().view(b_size, -1)

        memory = self.mac(lstm_out, h, img)

        embed_a0 = self.embed(answer_embeddings[0])
        embed_a1 = self.embed(answer_embeddings[1])
        embed_a2 = self.embed(answer_embeddings[2])
        embed_a3 = self.embed(answer_embeddings[3])
        embed_a4 = self.embed(answer_embeddings[4])

        # answers = self.embed(answer_embeddings)
        answers = torch.zeros(b_size, self.classes, self.opt.emb_dim).to(device)

        # take average for each answer
        answers[:, 0,] = torch.sum(embed_a0, dim=1) / answer_len[:, 0].view(b_size, 1).float() #check dim
        answers[:, 1,] = torch.sum(embed_a1, dim=1) / answer_len[:, 1].view(b_size, 1).float()
        answers[:, 2,] = torch.sum(embed_a2, dim=1) / answer_len[:, 2].view(b_size, 1).float()
        answers[:, 3,] = torch.sum(embed_a3, dim=1) / answer_len[:, 3].view(b_size, 1).float()
        answers[:, 4,] = torch.sum(embed_a4, dim=1) / answer_len[:, 4].view(b_size, 1).float()

        # answers = answers.view(b_size, 5, -1)

        memory = self.final_memory(memory)
        ans_0 = self.linear(answers[:, 0,])
        ans_1 = self.linear(answers[:, 1,])
        ans_2 = self.linear(answers[:, 2,])
        ans_3 = self.linear(answers[:, 3,])
        ans_4 = self.linear(answers[:, 4,])

        similarities = torch.zeros(b_size,
                                   self.classes)  # (num_units, b_size, 5) if computing similarity b/w every memory write and answers

        if torch.cuda.is_available():
            similarities = similarities.cuda()

            # calculate cosine similarity between answer feature and memory
            similarities[:, 0] = self.cos(memory.double(), ans_0.double())
            similarities[:, 1] = self.cos(memory.double(), ans_1.double())
            similarities[:, 2] = self.cos(memory.double(), ans_2.double())
            similarities[:, 3] = self.cos(memory.double(), ans_3.double())
            similarities[:, 4] = self.cos(memory.double(), ans_4.double())

        out = torch.cat([memory, h, similarities], 1)
        out = self.classifier(out)

        # print("\tIn Model: input size", image.size(),
        #       "output size", out.size())

        return out
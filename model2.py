import gc

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import kaiming_uniform_, xavier_uniform_

from rnn import RNNEncoder
from utils import masked_softmax


def linear(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()

    return lin


def weights_init(m):
    if isinstance(m, nn.Conv3d):
        kaiming_uniform_(m.weight.data)
        m.bias.data.zero_()


class ControlUnit(nn.Module):
    def __init__(self, dim, max_step):
        super(ControlUnit, self).__init__()

        self.position_aware = nn.ModuleList()
        for i in range(max_step):
            self.position_aware.append(linear(dim, dim))

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
    def __init__(self, dim, max_step):
        super(ReadUnit, self).__init__()

        self.mem = linear(dim, dim)
        # self.knowlegde = nn.ModuleList()
        # for i in range(max_step):
        #     self.knowlegde.append(linear(dim, dim))
        self.knowlegde = linear(dim, dim)
        self.concat = linear(dim * 2, dim)
        self.attn = linear(dim, 1)
        self.dim = dim

    def forward(self, memory, know, control, num_units, step=None):
        mem = self.mem(memory)
        # position_aware = self.position_aware[step](question)
        know = self.knowlegde(know)

        concat = self.concat(torch.cat([mem * know, know], 2))
        control = control[-1].unsqueeze(1).repeat(1, num_units, 1)#.view(-1, self.dim)
        # attn = concat * control
        # attn = self.attn(attn)
        # attn = F.softmax(attn, 1)
        #
        # read = (attn * know).sum(1)
        read = concat * control

        return read

class SimpleReadUnit(nn.Module):
    def __init__(self, dim, max_steps=None):
        super(SimpleReadUnit, self).__init__()

        self.attn = linear(dim, 1)
        self.dim = dim

    def forward(self, memory, know, control, num_units, step=None):

        control = control[-1].unsqueeze(1).repeat(1, num_units, 1)

        control_know = (control * know)

        # know_attn_weight = self.attn(control_know)
        #
        # attn = F.softmax(know_attn_weight, 1)
        # read = (attn * control_know).sum(1)
        read = control_know
        return read


class ConvUnit(nn.Module):
    def __init__(self, dim, pretrained=False):
        super(ConvUnit, self).__init__()

        self.pretrained = pretrained

        self.conv = nn.Sequential(nn.Conv3d(1024, dim, 3, padding=1),
                                  # nn.BatchNorm3d(dim),
                                  nn.ELU(),
                                  nn.Conv3d(dim, dim, 3, padding=1),
                                  # nn.BatchNorm3d(dim),
                                  nn.ELU(),
                                  )
        #conv3d
        kaiming_uniform_(self.conv[0].weight)
        self.conv[0].bias.data.zero_()
        # batch norm
        # self.conv[1].weight.data.fill_(1)
        # self.conv[1].bias.data.zero_()
        #conv3d
        kaiming_uniform_(self.conv[2].weight)
        self.conv[2].bias.data.zero_()
        #batch norm
        # self.conv[4].weight.data.fill_(1)
        # self.conv[4].bias.data.zero_()

    # this function will be removed in future, not in use in current code
    # def load_pretrained_weights(self):
    #     if self.pretrained:
    #         c3d = C3D(num_classes=101, pretrained=True)
    #         self.conv[2].weight.data.copy_(c3d.conv5b.weight.data)
    #         self.conv[2].bias.data.copy_(c3d.conv5b.bias.data)

    def forward(self, x):
        return self.conv(x)


class ListModule(nn.Module):
    def __init__(self, dim, module, num_units=10, self_attention=False, memory_gate=False, shared=True):

        super(ListModule, self).__init__()
        self.modules_allowed = ['read', 'write', 'conv3d']
        # self.module_dict = nn.ModuleDict([
        #     ['read', ReadUnit(dim)],
        #     ['write', WriteUnit(dim, self_attention, memory_gate)],
        #     ['conv3d', ConvUnit(dim)]
        #     #TODO: add conv3D in dict and support in forward()
        # ])
        self.module_name = module
        self.module = self.get_module(dim, module, self_attention, memory_gate)
        self.shared = shared
        self.num_units = num_units
        self.dim = dim


        if shared:
            num_units = 1

        self.module_list = nn.ModuleList([self.module for i in range(num_units)])
        # self.outputs = torch.zeros((2, 2))  # output shape

    def get_module(self, dim, module_name, self_attn=False, mem_gate=False):
        module = None

        if module_name not in self.modules_allowed:
            raise ValueError(
                "Incorrect module name. You gave: {}, allowed modules are {}".format(module_name, self.modules_allowed))
        if module_name == 'read':
            module = ReadUnit(dim)
        elif module_name == 'write':
            module = WriteUnit(dim, self_attn, mem_gate)
        elif module_name == 'conv3d':
            module = ConvUnit(dim)

        return module

    def forward(self, x, memory=None, control=None, num_blocks=None):
        out = []

        # ModuleList can act as an iterable, or be indexed using ints
        if self.shared and num_blocks is not None:
            layer = self.module_list[0]
            if isinstance(layer, ConvUnit):
                bsz, num_blocks, num_feat, t, h, w = x.size()

                # flatten x in batchxunits dim
                x = x.view(bsz * num_blocks, num_feat, t, h, w)
                out = layer(x.float()).view(bsz, num_blocks, self.dim, t, h, w)
            elif isinstance(layer, ReadUnit):
                bsz, num_blocks, dim1, dim2 = x.size()

                x = x.view(bsz * num_blocks, dim1, dim2)
                # reshape memory to be passed through layer in one go
                # mem = memory[-1]  # blk, bsz, dim -> bsz, blk, dim
                mem = memory[-1].contiguous().view(bsz * num_blocks, dim1)

                # reshape control signal
                control = [con.unsqueeze(1).repeat(1, num_blocks, 1).view(-1, dim1) for con in control]
                #add a check to see that reshaping doesnot destroy the correct order
                out = layer(mem, x.float(), control).view(bsz, num_blocks, -1)

            elif isinstance(layer, WriteUnit):
                bsz, num_blocks, dim = x.squeeze().size()
                x = x.view(bsz*num_blocks, dim)

                # mem = [torch.stack(mem).permute(1, 0, 2) for mem in memory] #blk, bsz, dim -> bsz, blk, dim
                mem = [mem.contiguous().view(bsz * num_blocks, dim) for mem in memory]

                # reshape control signal
                control = [con.unsqueeze(1).repeat(1, num_blocks, 1).view(-1, dim) for con in control]
                # memories_at_index = [memory[m][index] for m in range(len(memory))]
                out = layer(mem, x.float(), control).view(bsz, num_blocks, -1)

        else:
            for index, layer in enumerate(self.module_list):
                if isinstance(layer, ConvUnit):
                    bsz, num_blocks, num_feat, t, h, w = x.size()

                    # flatten x in batchxunits dim
                    x = x.view(bsz * num_blocks, num_feat, t, h, w)
                    out.append(layer(x.float()).view(bsz, num_blocks, self.dim, t, h, w))

                elif isinstance(layer, ReadUnit):
                    bsz, num_blocks, dim1, dim2 = x.size()
                    x = x.view(bsz * num_blocks, dim1, dim2)
                    # reshape memory to be passed through layer in one go
                    mem = torch.stack(memory[-1]).permute(1, 0, 2)  # blk, bsz, dim -> bsz, blk, dim
                    mem = mem.contiguous().view(bsz * num_blocks, dim1)

                    # reshape control signal
                    control = [con.unsqueeze(1).repeat(1, num_blocks, 1).view(-1, dim1) for con in control]
                    # add a check to see that reshaping doesnot destroy the correct order
                    out.append(layer(mem, x.float(), control).view(bsz, num_blocks, -1))

                elif isinstance(layer, WriteUnit):
                    bsz, num_blocks, dim = x.squeeze().size()
                    x = x.view(bsz * num_blocks, dim)

                    mem = [torch.stack(mem).permute(1, 0, 2) for mem in memory]  # blk, bsz, dim -> bsz, blk, dim
                    mem = [mem.contiguous().view(bsz * num_blocks, dim) for mem in mem]

                    # reshape control signal
                    control = [con.unsqueeze(1).repeat(1, num_blocks, 1).view(-1, dim) for con in control]
                    # memories_at_index = [memory[m][index] for m in range(len(memory))]

                    out.append(layer(mem, x.float(), control).view(bsz, num_blocks, -1))

        if not self.shared:
            outputs = torch.stack(out)
        else:
            outputs = out

        return outputs


class WriteUnit(nn.Module):
    def __init__(self, dim, self_attention=False, memory_gate=False):
        super(WriteUnit, self).__init__()

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
        concat = self.concat(torch.cat([retrieved, prev_mem], 2))
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
                 dropout=0.15, shared=True):
        super(MACUnit, self).__init__()

        self.control = ControlUnit(dim, max_step)
        self.read_units = ReadUnit(dim, max_step)
        self.write_units = WriteUnit(dim, self_attention=False, memory_gate=False)
        # self.read_units = ListModule(dim, 'read', shared)  # list of read units
        # self.write_units = ListModule(dim, 'write', shared)  # list of write units

        self.mem_0 = nn.Parameter(torch.zeros(1, 1, dim))  # bsz, num_units, dim
        self.control_0 = nn.Parameter(torch.zeros(1, dim))  # bsz, dim
        self.linear = nn.Linear(dim, 1)

        self.dim = dim
        self.max_step = max_step
        self.dropout = dropout

    def get_mask(self, x, dropout):
        mask = torch.empty_like(x).bernoulli_(1 - dropout)
        mask = mask / (1 - dropout)

        return mask

    def forward(self, context, question, knowledge, know_masks=None, num_units=4):
        """

        :param context:
        :param question: (bsz, max(q_len), dim)
        :param knowledge: (bsz*num_blocks, dim, flattened_dim(t*h*w))
        :param num_units: (max(num_blocks in batch))
        :return:
        """
        b_size = question.size(0)

        control = self.control_0.expand(b_size, self.dim)
        memory = self.mem_0.expand(b_size, num_units, self.dim)  # bsz, num_units, dim

        if self.training:
            control_mask = self.get_mask(control, self.dropout)
            memory_masks = self.get_mask(memory.view(b_size * num_units, self.dim), self.dropout) # bsz x num_units x dim
            # memory_masks = [self.get_mask(mem, self.dropout) for mem in memory]
            control = control * control_mask
            memory = memory.view(b_size * num_units, -1) * memory_masks
            memory = memory.view(b_size, num_units, self.dim)
            # memory = [memory[i] * memory_masks[i] for i in range(num_units)]

        controls = [control]
        memories = [memory]
        clips_attn = []
        for step in range(self.max_step):
            control = self.control(step, context, question, control)
            if self.training:
                control = control * control_mask
            controls.append(control)
            _, _, dim1 = knowledge.size()

            # reshape memory to be passed through layer in one go
            mem = memories[-1]#.contiguous().view(b_size * num_units, dim1)
            # reshape control signal
            #controls = [con.unsqueeze(1).repeat(1, num_units, 1).view(-1, dim1) for con in controls]
            reads = self.read_units(mem, knowledge, controls, num_units, step)  # reads for each temporal block
            mem = [mem for mem in memories]
            memory = self.write_units(mem, reads.float(), controls).view(-1, dim1)  # writes for each temporal block

            if self.training:
                memory = memory * memory_masks
            memory = memory.view(b_size, num_units, self.dim)

            clip_attn = self.linear(memory)
            if know_masks is not None:
                clip_attn = masked_softmax(clip_attn.squeeze(2), know_masks, dim=1) # --> (bsz, num_units, 1)
            else:
                clip_attn = nn.Softmax(clip_attn, dim=1)

            # clips_attn.append(clip_attn)
            memories.append(memory)

        return memory, clip_attn, memories

class SimpleMACUnit(nn.Module):
    def __init__(self, dim, max_step=12,
                 self_attention=False, memory_gate=False,
                 dropout=0.15, shared=True, variant=1):
        super(SimpleMACUnit, self).__init__()

        self.control = ControlUnit(dim, max_step)
        if variant == 1:
            self.read_units = ReadUnit(dim, max_step)
        elif variant == 2:
            self.read_units = SimpleReadUnit(dim, max_step)
        else:
            assert variant in [1, 2], \
                "Incorrect variant number for ReadUnit. Valid options are 1: ReadUnit, 2: SimpleReadUnit"

        self.mem_0 = nn.Parameter(torch.zeros(1, 1, dim))  # bsz, num_units, dim
        self.control_0 = nn.Parameter(torch.zeros(1, dim))  # bsz, dim
        self.linear = nn.Linear(dim, 1)

        self.dim = dim
        self.max_step = max_step
        self.dropout = dropout

    def get_mask(self, x, dropout):
        mask = torch.empty_like(x).bernoulli_(1 - dropout)
        mask = mask / (1 - dropout)

        return mask

    def forward(self, context, question, knowledge, know_masks=None, num_units=4, clip_attn=True):
        """

        :param context:
        :param question: (bsz, max(q_len), dim)
        :param knowledge: (bsz*num_blocks, dim, flattened_dim(t*h*w))
        :param num_units: (max(num_blocks in batch))
        :return:
        """
        b_size = question.size(0)

        control = self.control_0.expand(b_size, self.dim)
        memory = self.mem_0.expand(b_size, num_units, self.dim)  # bsz, num_units, dim

        if self.training:
            control_mask = self.get_mask(control, self.dropout)
            memory_masks = self.get_mask(memory.view(b_size * num_units, self.dim), self.dropout) # bsz x num_units x dim
            # memory_masks = [self.get_mask(mem, self.dropout) for mem in memory]
            control = control * control_mask
            memory = memory.view(b_size * num_units, -1) * memory_masks
            memory = memory.view(b_size, num_units, self.dim)

        controls = [control]
        memories = [memory]
        clips_attn = []
        for step in range(self.max_step):
            control = self.control(step, context, question, control)
            if self.training:
                control = control * control_mask
            controls.append(control)
            _, _, dim1 = knowledge.size()

            # reshape memory to be passed through layer in one go
            mem = memories[-1]#.contiguous().view(b_size * num_units, dim1)
            memory = self.read_units(mem, knowledge, controls, num_units, step)  # reads for each temporal block

            if self.training:
                memory = memory.view(b_size * num_units, -1) * memory_masks
            memory = memory.view(b_size, num_units, self.dim)

            clip_attn = self.linear(memory)
            if know_masks is not None:
                clip_attn = masked_softmax(clip_attn.squeeze(2), know_masks, dim=1) # --> (bsz, num_units, 1)
            else:
                clip_attn = nn.Softmax(clip_attn, dim=1)

            # clips_attn.append(clip_attn)
            memories.append(memory)

        return memory, clip_attn, memories


class MACNetwork(nn.Module):
    def __init__(self, n_vocab, dim, opt, embed_hidden=300,
                 max_step=12, self_attention=False, memory_gate=False,
                 classes=5, dropout=0.15, shared=True):
        super(MACNetwork, self).__init__()

        self.conv = ListModule(dim, 'conv3d', shared)

        self.embed = nn.Embedding(n_vocab, embed_hidden)
        # self.lstm = nn.LSTM(embed_hidden, dim,
        #                     batch_first=True, bidirectional=True)

        self.lstm_a = RNNEncoder(embed_hidden, dim, True, dropout_p=0, n_layers=1, rnn_type="lstm")
        self.lstm_proj = nn.Linear(dim * 2, dim)

        # self.bidaf = BidafAttn(embed_hidden * 3, method="dot")  # no parameter for dot

        self.mac = MACUnit(dim, max_step,
                           self_attention, memory_gate, dropout, shared)
        # self.linear = linear(dim,1)

        # self.answers = nn.Linear(1024,dim).double()
        self.cos = nn.CosineSimilarity(eps=1e-6)

        self.classifier = nn.Sequential(linear((dim*3+classes), dim),
                                        nn.ELU(),
                                        # nn.Dropout(p=0.25),
                                        linear(dim, classes))

        self.max_step = max_step
        self.dim = dim
        self.opt = opt

        self.reset()

    def reset(self):
        if not self.opt.no_glove:
            self.load_embedding(self.opt.vocab_embedding)
        else:
            self.embed.weight.data.uniform_(0, 1)

        kaiming_uniform_(self.classifier[0].weight)

    def forward(self, image, question, question_len, answer_embeddings, answer_len, dropout=0.15, num_units=10):
        b_size = question.size(0)

        img = self.conv(image, num_blocks=num_units)

        # #check if it is the right way to swap axis
        img = img.view(b_size, num_units, self.dim, -1)

        embed = self.embed(question)
        # embed = nn.utils.rnn.pack_padded_sequence(embed, question_len,
        #                                           batch_first=True)
        # lstm_out, (h, _) = self.lstm(embed)
        # lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out,
        #                                                batch_first=True)

        lstm_out, h = self.lstm_a(embed, torch.as_tensor(question_len))

        lstm_out = self.lstm_proj(lstm_out)

        # h = h.view(b_size, -1)

        embed_a0 = self.embed(answer_embeddings[0])
        embed_a1 = self.embed(answer_embeddings[1])
        embed_a2 = self.embed(answer_embeddings[2])
        embed_a3 = self.embed(answer_embeddings[3])
        embed_a4 = self.embed(answer_embeddings[4])

        a0_l, a1_l, a2_l, a3_l, a4_l = answer_len[0], answer_len[1], answer_len[2], answer_len[3], answer_len[4]

        lstm_out_a0, h0 = self.lstm_a(embed_a0, a0_l)
        lstm_out_a1, h1 = self.lstm_a(embed_a1, a1_l)
        lstm_out_a2, h2 = self.lstm_a(embed_a2, a2_l)
        lstm_out_a3, h3 = self.lstm_a(embed_a3, a3_l)
        lstm_out_a4, h4 = self.lstm_a(embed_a4, a4_l)

        # TODO: compute answer-aware video representation and send to mac network.
        # lstm_out_a0 = self.lstm_proj(lstm_out_a0)
        # lstm_out_a1 = self.lstm_proj(lstm_out_a1)
        # lstm_out_a2 = self.lstm_proj(lstm_out_a2)
        # lstm_out_a3 = self.lstm_proj(lstm_out_a3)
        # lstm_out_a4 = self.lstm_proj(lstm_out_a4)
        # ....
        # context: answer, query: video
        # v_a0, _ = self.bidaf(lstm_out_a0, a0_l, img, num_units)
        # v_a1, _ = self.bidaf(lstm_out_a1, a1_l, img, num_units)
        # v_a2, _ = self.bidaf(lstm_out_a2, a2_l, img, num_units)
        # v_a3, _ = self.bidaf(lstm_out_a3, a3_l, img, num_units)
        # v_a4, _ = self.bidaf(lstm_out_a4, a4_l, img, num_units)
        # try to combine these attended representations into one and see if sending them to mac helps.
        # generate 5 to 1 attention on these v_a0,...,v_a4 and compute weighted sum (or average)

        h0 = self.lstm_proj(h0)
        h1 = self.lstm_proj(h1)
        h2 = self.lstm_proj(h2)
        h3 = self.lstm_proj(h3)
        h4 = self.lstm_proj(h4)

        ans_concat=torch.cat([h0.unsqueeze(2), h1.unsqueeze(2), h2.unsqueeze(2), h4.unsqueeze(2), h4.unsqueeze(2)],2).unsqueeze(1)
        ans_concat = ans_concat.expand(b_size, num_units, self.dim, 5)
        img = torch.cat([img, ans_concat], 3)

        memory, clip_attn, mem_matrix = self.mac(lstm_out, h, img)

        if self.training:
            memory = torch.sum(torch.stack(memory) * clip_attn, dim=0)  # (b_size,512)
        else:
            memory = torch.sum(memory * clip_attn, dim=0)  # (b_size,512)

        similarities = torch.zeros(b_size,
                                   5)  # (num_units, b_size, 5) if computing similarity b/w every memory write and answers

        if torch.cuda.is_available():
            similarities = similarities.cuda()

        # calculate cosine similarity between answer feature and memory
        similarities[:, 0] = self.cos(memory.double(), h0.double())
        similarities[:, 1] = self.cos(memory.double(), h1.double())
        similarities[:, 2] = self.cos(memory.double(), h2.double())
        similarities[:, 3] = self.cos(memory.double(), h3.double())
        similarities[:, 4] = self.cos(memory.double(), h4.double())

        ######### for each clip without collapsing them into one weighted representation #########
        # for i in range(np.size(answer_embeddings, axis=1)):
        #     for j in range(len(memory)):
        #         similarities[j, :, i] = self.cos(memory[j].double(), answer_embeddings[:, i, :].double())

        # TODO: Add normalization/attention code for memory reads \\
        # select memory based on the highest attention and send to classifier

        out = torch.cat([memory, h, similarities], 1)
        out = self.classifier(out)
        gc.collect()
        # del img, lstm_out, h, similarities, answers, memory

        return out

    def load_embedding(self, pretrained_embedding):
        self.embed.weight.data.copy_(torch.from_numpy(pretrained_embedding))

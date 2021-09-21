import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_, xavier_uniform_
from torch.autograd import Variable
# from models.rnn import RNNEncoder
from bidaf import BidafAttn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def linear(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()

    return lin

class ControlUnit(nn.Module):
    def __init__(self, dim, max_step):
        super(ControlUnit, self).__init__()

        self.position_aware = nn.ModuleList()
        for i in range(max_step):
            self.position_aware.append(linear(dim, dim))

        self.control_question = linear(dim * 2, dim)
        self.attn = linear(dim, 1)
        self.attn_dropout = nn.Dropout(0.2)

        self.dim = dim

    def forward(self, step, context, question, control, prev_mem=None):
        position_aware = self.position_aware[step](question)

        control_question = torch.cat([control, position_aware], 1)
        control_question = self.control_question(control_question)
        control_question = control_question.unsqueeze(1)

        context_prod = control_question * context
        attn_weight = self.attn(context_prod)
        attn_weight = self.attn_dropout(attn_weight)

        attn = F.softmax(attn_weight, 1)

        next_control = (attn * context).sum(1)

        return next_control, attn

class SimpleControlUnit(nn.Module):
    def __init__(self, dim):
        super(SimpleControlUnit, self).__init__()

        self.position_aware = linear(dim, dim)


        self.control_question = linear(dim * 2, dim)
        self.attn = linear(dim, 1)
        self.attn_dropout = nn.Dropout(0.2)

        self.dim = dim

    def forward(self, context, question):
        position_aware = self.position_aware(question)

        # control_question = torch.cat([control, position_aware], 1)
        # control_question = self.control_question(control_question)
        control_question = position_aware.unsqueeze(1)

        context_prod = control_question * context
        attn_weight = self.attn(context_prod)
        attn_weight = self.attn_dropout(attn_weight)

        attn = F.softmax(attn_weight, 1)

        next_control = (attn * context).sum(1)

        return next_control, attn

class SelfAttentionUnit(nn.Module):
    def __init__(self, dim, method="dot"):
        super(SelfAttentionUnit, self).__init__()
        self.method = method
        self.proj1 = linear(dim, dim)
        self.proj2 = linear(dim, dim)

        self.bn1 = nn.Sequential(
                        nn.BatchNorm1d(dim, affine=True),
                        nn.Tanh()
                    )

        self.bn2 = nn.Sequential(
                        nn.BatchNorm1d(dim, affine=True),
                        nn.Tanh()
                    )

        self.reduction1 = linear(dim, 16)
        self.reduction2 = linear(dim, 16)

        self.fc = linear(dim, 1)
        self.dim = dim

    def similarity(self, s1, l1, s2, l2):
        """
        :param s1: [B, t1, D]
        :param l1: [B]
        :param s2: [B, t2, D]
        :param l2: [B]
        :return:
        """
        if self.method == "original":
            t1 = s1.size(1)
            t2 = s2.size(1)
            repeat_s1 = s1.unsqueeze(2).repeat(1, 1, t2, 1)  # [B, T1, T2, D]
            repeat_s2 = s2.unsqueeze(1).repeat(1, t1, 1, 1)  # [B, T1, T2, D]
            packed_s1_s2 = torch.cat([repeat_s1, repeat_s2, repeat_s1 * repeat_s2], dim=3)  # [B, T1, T2, D*3]
            s = self.mlp(packed_s1_s2).squeeze()  # s is the similarity matrix from biDAF paper. [B, T1, T2]
        elif self.method == "dot":
            s = torch.bmm(s1, s2.transpose(1, 2))

        s_mask = s.data.new(*s.size()).fill_(1).byte()  # [B, T1, T2]
        # Init similarity mask using lengths
        for i, (l_1, l_2) in enumerate(zip(l1, l2)):
            s_mask[i][:l_1, :l_2] = 0

        s_mask = Variable(s_mask)
        s.data.masked_fill_(s_mask.data.byte(), -float("inf"))
        return s

    def tile_concat(self, in1, in2):
        assert (in1.shape == in2.shape)
        b, n, d = in1.shape
        t1 = in1.unsqueeze(2).repeat(1, 1, n, 1)
        t2 = in2.unsqueeze(1).repeat(1, n, 1, 1)
        out = torch.cat([t1, t2], -1)
        return out

    def tile_concat2(self, in1, in2):
        assert (in1.shape == in2.shape)
        b, n, d = in1.shape
        out = torch.zeros(b, n, n, d*2)

        for i in range(n):
            for j in range(n):
                out[:, i, j] = torch.cat([in1[:, i], in2[:, j]], dim=-1)

        return out

    def tile_sum(self, in1, in2):
        assert (in1.shape == in2.shape)
        b, n, d = in1.shape

        #in1 = self.reduction1(in1)
        #in2 = self.reduction2(in2)
        know_proj2_t = torch.transpose(in2, -2, -1)  # transpose last two dimensions

        prod = torch.bmm(in1, know_proj2_t)
        # t1 = in1.unsqueeze(2).repeat(1, 1, n, 1)
        # t2 = in2.unsqueeze(1).repeat(1, n, 1, 1)
        # out = t1 + t2

        return prod

    def forward(self, knowledge, know_l):
        max_len = max(know_l)
        know_masks = torch.arange(max_len).expand(len(know_l), max_len).to('cuda') < know_l.unsqueeze(1)


        know_proj1 = self.proj1(knowledge)
        know_proj2 = self.proj2(knowledge)

        # know_proj1 = self.bn1(know_proj1.permute(0, 2, 1)).permute(0, 2, 1)
        # know_proj2 = self.bn2(know_proj2.permute(0, 2, 1)).permute(0, 2, 1)

        # tiled_know = self.tile_sum(know_proj1, know_proj2) # ==> (N, N, 2d)
        # scores = self.fc(tiled_know)
        scores = self.similarity(know_proj1, know_l, know_proj2, know_l)
        # masked_scores = tiled_know.masked_fill(know_masks.unsqueeze(1) == 0, -1e9)
        # know_proj2_t = torch.transpose(know_proj2, -2, -1)  #transpose last two dimensions
        # prod = torch.bmm(know_proj1, know_proj2_t)
        soft_weights = nn.functional.softmax(scores, -1)
        soft_weights.data.masked_fill_(soft_weights.data != soft_weights.data, 0)  # remove nan from softmax on -inf
        weighted_knowledge = torch.bmm(soft_weights, knowledge)
        return weighted_knowledge


class TwoLayerSelfAttention(nn.Module):
    def __init__(self, dim):
        super(TwoLayerSelfAttention, self).__init__()

        self.set_attn1 = SelfAttentionUnit(dim)
        self.set_attn2 = SelfAttentionUnit(dim)
        self.bn1 = nn.BatchNorm1d(dim)



    def forward(self, x, x_l):
        x = self.set_attn1(x, x_l)
        # x = self.bn1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.set_attn2(x, x_l)
        return x

class MaskedControlUnit(nn.Module):
    def __init__(self, dim, max_step):
        super(MaskedControlUnit, self).__init__()

        self.position_aware = nn.ModuleList()
        for i in range(max_step):
            self.position_aware.append(linear(dim, dim))
            # self.position_aware.append(
            #     nn.Sequential(
            #         nn.LayerNorm(dim),
            #         # nn.Dropout(0.2),
            #         linear(dim, dim),
            #         nn.ReLU(),
            #         nn.LayerNorm(dim)
            #     )
            # )
        self.ln1 = nn.Sequential(
                        nn.LayerNorm(dim),
                        nn.ReLU()
                    )
        self.control_question = linear(dim * 3, dim)
        self.ln2 = nn.Sequential(
                        nn.LayerNorm(dim),
                        nn.ReLU()
                    )
        # self.control_question = nn.Sequential(nn.LayerNorm(dim*3),
        #                                       # nn.Dropout(0.2),
        #                                       linear(dim * 3, dim),
        #                                       nn.ReLU(),
        #                                       nn.LayerNorm(dim)
        #                                       )
        self.attn = linear(dim, 1)
        # self.bn3 = nn.Sequential(
        #     nn.BatchNorm1d(dim),
        #     nn.ReLU()
        # )
        self.attn_dropout = nn.Dropout(0.5)

        self.context_mask_fusion = linear(dim * 2, dim)
        self.ln3 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU()
        )
        self.ln4 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU()
        )

        self.dim = dim

    def forward(self, step, context, question, control, qm1, qm_l1, qm2, qm_l2, prev_mem=None):
        qm1 = qm1.float()
        qm2 = qm2.float()
        qm = qm1+qm2
        position_aware = self.position_aware[step](question)
        position_aware = self.ln1(position_aware)

        control_question = torch.cat([control, position_aware], 1)
        control_question = self.control_question(control_question)
        control_question = self.ln2(control_question)
        control_question = control_question.unsqueeze(1)

        context_prod = control_question * context
        attn_weight = self.attn(context_prod)/math.sqrt(self.dim)
        attn_weight = self.attn_dropout(attn_weight)

        attn_weight = attn_weight.masked_fill(qm.unsqueeze(2) == 0, -1e9) #put -inf on positions where mask = 0 b/c softmax(-inf)=0
        attn = F.softmax(attn_weight, 1)

        #apply mask on context words
        masked_ctx1 = context * qm1.unsqueeze(2)
        masked_ctx2 = context * qm2.unsqueeze(2)

        # element-wise product between masked context and control_question
        masked_ctx_prod1 = control_question * masked_ctx1
        masked_ctx_prod2 = control_question * masked_ctx2

        attn_weight1 = self.attn(masked_ctx_prod1)/math.sqrt(self.dim)
        attn_weight1 = self.attn_dropout(attn_weight1)
        attn_weight1 = attn_weight1.masked_fill(qm1.unsqueeze(2) == 0, -1e9) #put -inf on positions where mask = 0 b/c softmax(-inf)=0
        attn1 = F.softmax(attn_weight1, 1)

        attn_weight2 = self.attn(masked_ctx_prod2)/math.sqrt(self.dim)
        attn_weight2 = self.attn_dropout(attn_weight2)
        attn_weight2 = attn_weight2.masked_fill(qm2.unsqueeze(2) == 0, -1e9)
        attn2 = F.softmax(attn_weight2, 1)

        masked_control1 = F.normalize((attn1 * masked_ctx1).sum(1))
        masked_control2 = F.normalize((attn2 * masked_ctx2).sum(1))

        next_control_full = F.normalize((attn * context).sum(1))
        # masked_control1 = (attn1 * masked_ctx1).sum(1)
        # masked_control2 = (attn2 * masked_ctx2).sum(1)
        #
        # next_control_full = (attn * context).sum(1)

        fused_q1 = torch.cat([masked_control1, next_control_full], 1)
        fused_q2 = torch.cat([masked_control2, next_control_full], 1)

        ctrl1 = self.context_mask_fusion(fused_q1)
        ctrl1 = self.ln3(ctrl1)

        ctrl2 = self.context_mask_fusion(fused_q2)
        ctrl2 = self.ln4(ctrl2)

        next_control = torch.cat([ctrl1, ctrl2], 1)
        attentions = torch.cat([attn[:, None, :], attn1[:, None, :], attn2[:, None, :]], 1) #B, 3, Q_l
        return next_control, attentions

class ForgetGate(nn.Module):
    def __init__(self, dim):
        super(ForgetGate, self).__init__()

        self.in_layer1 = linear(dim, dim)
        self.in_layer2 = linear(dim, dim)
        self.dim = dim

    def forward(self, previous, current):
        out = self.in_layer1(previous) + self.in_layer2(current)
        forget = torch.sigmoid(out) * previous
        retained = forget + current
        return retained

class ReadUnit(nn.Module):
    def __init__(self, dim):
        super(ReadUnit, self).__init__()

        self.mem = linear(dim, dim)
        self.concat = linear(dim * 2, dim)
        self.attn = linear(dim, 1)

    def forward(self, memory, know, control):
        mem = self.mem(memory[-1]).unsqueeze(1)
        concat = self.concat(torch.cat([mem * know, know], 2))

        attn = concat * control[-1].unsqueeze(1)
        attn = self.attn(attn)
        attn = F.softmax(attn, 1)

        read = (attn * know).sum(1)

        return read


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
    def __init__(self, dim, num_layers, max_step=12,
                self_attention=False, memory_gate=False,
                dropout=0.15, q_only=False):
        super(MACUnit, self).__init__()

        self.control = MaskedControlUnit(dim, max_step)

        self.mem_0 = nn.Parameter(torch.zeros(1, dim*2))
        self.control_0 = nn.Parameter(torch.zeros(1, dim*2))
        self.attn = linear(dim, 1)
        self.attn2 = linear(dim, 1)

        # num_layers = 2
        # layers = []
        # for i in range(num_layers):
        #     layers.append(SelfAttentionUnit(dim))
        #
        # self.setattn = nn.Sequential(*layers)
        # self.proj1 = linear(dim, dim)
        # self.proj2 = linear(dim, dim)

        self.setattn = BidafAttn(dim, "dot")

        self.attn_dropout = nn.Dropout(0.2)

        self.cos = nn.CosineSimilarity(dim=2)
        self.dim = dim
        self.max_step = max_step
        self.dropout = dropout
        self.q_only = q_only

    def get_mask(self, x, dropout):
        mask = torch.empty_like(x).bernoulli_(1 - dropout)
        mask = mask / (1 - dropout)

        return mask

    def forward(self, context, question, knowledge, know_l, qm1, qm_l1, qm2, qm_l2):
        b_size = question.size(0)
        max_len = max(know_l)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # know_masks = torch.arange(max_len).expand(len(know_l), max_len).to(device) < know_l.unsqueeze(1)

        control = self.control_0.expand(b_size, self.dim*2)
        memory = self.mem_0.expand(b_size, self.dim*2)

        if self.training:
            control_mask = self.get_mask(control, self.dropout)
            memory_mask = self.get_mask(memory, self.dropout)

            control = control * control_mask
            memory = memory * memory_mask

        controls = [control]
        memories = [memory]
        q_attns, attns = [], []
        #attend feats which weren't attended first time more
        # attn = torch.zeros(knowledge.size(1), 1).cuda()
        for i in range(self.max_step):
            prev_mem = memories[-1]
            control, q_attn = self.control(i, context, question, control, qm1, qm_l1, qm2, qm_l2)

            if self.training:
                control = control * control_mask

            # control = F.normalize(control, dim=1)
            control1, control2 = torch.split(control, self.dim, 1)

            # if self.training:
            #     control1 = control1 * control_mask
            #     control2 = control2 * control_mask
            controls.append(control)

            ####original implementation#####
            #use self attention on knowledge before using control on it
            # q_aware_know = question.unsqueeze(1) * knowledge
            # proj1 = self.proj1(knowledge)
            # proj2 = self.proj2(knowledge)
            # knowledge, _ = self.setattn(proj1, know_l, proj2, know_l)
            # knowledge = F.normalize(knowledge, dim=2)
            if self.q_only:
                return controls, None, None
            else:
                control_know1 = (control1.unsqueeze(1) * knowledge)
                control_know2 = (control2.unsqueeze(1) * knowledge)

                control_know1, _, _ = self.setattn(control_know1, know_l,control_know1, know_l)
                control_know2, _, _ = self.setattn(control_know2, know_l,control_know2, know_l)
                # attn = self.get_lookup_weighting(control_know, control, strengths=torch.ones(b_size, 1))
                # print(attn.size())
                know_attn_weight1 = self.attn(control_know1)/math.sqrt(self.dim)
                know_attn_weight1 = self.attn_dropout(know_attn_weight1)
                # know_attn_weight1 = know_attn_weight1.masked_fill(know_masks.unsqueeze(2) == 0, -1e9)
                attn1 = torch.softmax(know_attn_weight1, 1)

                know_attn_weight2 = self.attn(control_know2)/math.sqrt(self.dim)
                know_attn_weight2 = self.attn_dropout(know_attn_weight2)
                # know_attn_weight2 = know_attn_weight2.masked_fill(know_masks.unsqueeze(2) == 0, -1e9)
                attn2 = torch.softmax(know_attn_weight2, 1)

                attn = torch.cat([attn1[:, None, :], attn2[:, None, :]], 1)

                ##################################

                #####modified implementation with inner product####
                # attn = F.softmax(torch.bmm(control.unsqueeze(1).expand_as(knowledge).unsqueeze(2).contiguous().view(-1, 1, self.dim),
                #                            knowledge.view(-1, self.dim).unsqueeze(2)).squeeze(1).view(b_size, -1, 1), dim=1)
                ###################################################
                know_attn1 = attn1 * control_know1
                know_attn2 = attn2 * control_know2

                know_attn_weight1_lvl2 = self.attn2(know_attn1)/math.sqrt(self.dim)
                know_attn_weight1_lvl2 = self.attn_dropout(know_attn_weight1_lvl2)
                attn1_lvl2 = torch.softmax(know_attn_weight1_lvl2, 1)

                know_attn_weight2_lvl2 = self.attn2(know_attn2) / math.sqrt(self.dim)
                know_attn_weight2_lvl2 = self.attn_dropout(know_attn_weight2_lvl2)
                attn2_lvl2 = torch.softmax(know_attn_weight2_lvl2, 1)

                attn_lvl2 = torch.cat([attn1_lvl2[:, None, :], attn2_lvl2[:, None, :]], 1)
                memory1 = (attn1_lvl2 * know_attn1).sum(1) + (know_attn1).sum(1)
                memory2 = (attn2_lvl2 * know_attn2).sum(1) + (know_attn2).sum(1)
                ###################################################

                # know_attn = attn * control_know
                # attn2 = torch.softmax(self.attn2(know_attn), 1)
                # know_attn2 = attn2 * know_attn
                # attn3 = torch.softmax(self.attn3(know_attn2), 1)
                # know_attn3 = attn3 * know_attn2
                # attn4 = torch.softmax(self.attn4(know_attn3), 1)

                # memory1 = (attn1 * control_know1).sum(1)
                # memory2 = (attn2 * control_know2).sum(1)

                memory = torch.cat([memory1, memory2],1)

                # memory = self.forget_gate(prev_mem, memory)
                #read = self.read(memories, knowledge, controls)
                #memory = self.write(memories, read, controls)
                if self.training:
                    memory = memory * memory_mask
                memories.append(memory)
                # attns.append(attn)
                attns.append([attn, attn_lvl2])
                q_attns.append(q_attn)

            return memories, q_attns, attns

    def get_lookup_weighting(self, memory_matrix, keys, strengths):
        """
        retrives a content-based adderssing weighting given the keys
        Parameters:
        ----------
        memory_matrix: Tensor (batch_size, words_num, word_size)
            the memory matrix to lookup in
        keys: Tensor (batch_size, word_size, number_of_keys)
            the keys to query the memory with
        strengths: Tensor (batch_size, number_of_keys, )
            the list of strengths for each lookup key
        Returns: Tensor (batch_size, words_num, number_of_keys)
            The list of lookup weightings for each provided key
        """

        normalized_memory = F.normalize(memory_matrix, p=2, dim=2)
        normalized_keys = F.normalize(keys, p=2, dim=1)

        similiarity = self.cos(normalized_memory, normalized_keys.unsqueeze(1).repeat(1, normalized_memory.size(1), 1))
        # similiarity = torch.bmm(normalized_memory, normalized_keys.unsqueeze(2))
        strengths = strengths.to('cuda')

        return torch.softmax(similiarity * strengths, 1).unsqueeze(2)


class MACFusionUnit(nn.Module):
    def __init__(self, dim, max_step=12,
                self_attention=False, memory_gate=False,
                dropout=0.15):
        super(MACFusionUnit, self).__init__()

        self.control1 = ControlUnit(dim, max_step)
        self.control2 = ControlUnit(dim, max_step)
        #self.read = ReadUnit(dim)
        #self.write = WriteUnit(dim, self_attention, memory_gate)

        self.mem_0 = nn.Parameter(torch.zeros(1, dim))
        self.control_0_1 = nn.Parameter(torch.zeros(1, dim))
        self.control_0_2 = nn.Parameter(torch.zeros(1, dim))
        self.attn = linear(dim, 1)
        self.fusion = linear(dim*2, dim)
        # self.fusion = nn.Sequential(linear(dim*2, dim),
        #                             nn.ReLU(),
        #                             nn.Dropout(0.5)
        #                             )
        self.attn2 = linear(dim, 1)
        # self.attn3 = linear(dim, 1)
        # self.attn4 = linear(dim, 1)

        # self.forget_gate = ForgetGate(dim)

        self.dim = dim
        self.max_step = max_step
        self.dropout = dropout

    def get_mask(self, x, dropout):
        mask = torch.empty_like(x).bernoulli_(1 - dropout)
        mask = mask / (1 - dropout)

        return mask

    def forward(self, context, question, knowledge_src1, knowledge_src2):
        b_size = question.size(0)
        #control_know = torch.zeros_like(knowledge)

        control_1 = self.control_0_1.expand(b_size, self.dim)
        control_2 = self.control_0_2.expand(b_size, self.dim)

        memory = self.mem_0.expand(b_size, self.dim)

        if self.training:
            control_mask_1 = self.get_mask(control_1, self.dropout)
            control_mask_2 = self.get_mask(control_2, self.dropout)

            memory_mask = self.get_mask(memory, self.dropout)

            control_1 = control_1 * control_mask_1
            control_2 = control_2 * control_mask_2

            memory = memory * memory_mask

        controls1 = [control_1]
        controls2 = [control_2]
        memories = [memory]

        for i in range(self.max_step):
            # prev_mem = memories[-1]
            control1, q_attn = self.control1(i, context, question, control_1)
            control2, q_attn = self.control2(i, context, question, control_2)
            if self.training:
                control1 = control1 * control_mask_1
                control2 = control2 * control_mask_2
            controls1.append(control1)
            controls2.append(control2)
            ####original implementation#####
            attn1, control_know1 = self.control2knowledge(control1, knowledge_src1)
            attn2, control_know2 = self.control2knowledge(control2, knowledge_src2)

            ##################################

            #####modified implementation with inner product####
            # attn = F.softmax(torch.bmm(control.unsqueeze(1).expand_as(knowledge).unsqueeze(2).contiguous().view(-1, 1, self.dim),
            #                            knowledge.view(-1, self.dim).unsqueeze(2)).squeeze(1).view(b_size, -1, 1), dim=1)
            ###################################################
            # know_attn1 = attn1 * control_know1
            # know_attn2 = attn2 * control_know2
            # attn2_1 = torch.softmax(self.attn2(know_attn1), 1)
            # attn2_2 = torch.softmax(self.attn2(know_attn2), 1)
            # know_attn2 = attn2 * know_attn
            # attn3 = torch.softmax(self.attn3(know_attn2), 1)
            # know_attn3 = attn3 * know_attn2
            # attn4 = torch.softmax(self.attn4(know_attn3), 1)

            memory1 = (attn1 * control_know1).sum(1)
            memory2 = (attn2 * control_know2).sum(1)

            memory = torch.cat([memory1, memory2], dim=1)
            memory = self.fusion(memory)
            # memory = self.forget_gate(prev_mem, memory)
            #read = self.read(memories, knowledge, controls)
            #memory = self.write(memories, read, controls)
            if self.training:
                memory = memory * memory_mask
            memories.append(memory)

        return memory, q_attn, attn1, attn2

    def control2knowledge(self, control, knowledge_src1):
        control_know = control.unsqueeze(1) * knowledge_src1
        know_attn_weight = self.attn(control_know)
        attn = torch.softmax(know_attn_weight, 1)
        return attn, control_know


class MACUnitOriginal(nn.Module):
    def __init__(self, dim, max_step=12,
                self_attention=False, memory_gate=False,
                dropout=0.15):
        super(MACUnitOriginal, self).__init__()

        self.control = ControlUnit(dim, max_step)
        self.read = ReadUnit(dim)
        self.write = WriteUnit(dim, self_attention, memory_gate)

        self.mem_0 = nn.Parameter(torch.zeros(1, dim))
        self.control_0 = nn.Parameter(torch.zeros(1, dim))
        self.attn = linear(dim, 1)
        # self.attn2 = linear(dim, 1)
        # self.attn3 = linear(dim, 1)
        # self.attn4 = linear(dim, 1)

        # self.forget_gate = ForgetGate(dim)

        self.dim = dim
        self.max_step = max_step
        self.dropout = dropout

    def get_mask(self, x, dropout):
        mask = torch.empty_like(x).bernoulli_(1 - dropout)
        mask = mask / (1 - dropout)

        return mask

    def forward(self, context, question, knowledge):
        b_size = question.size(0)
        #control_know = torch.zeros_like(knowledge)

        control = self.control_0.expand(b_size, self.dim)
        memory = self.mem_0.expand(b_size, self.dim)

        if self.training:
            control_mask = self.get_mask(control, self.dropout)
            memory_mask = self.get_mask(memory, self.dropout)
            control = control * control_mask
            memory = memory * memory_mask

        controls = [control]
        memories = [memory]
        q_attns, attns = [], []
        #attend feats which weren't attended first time more
        attn = torch.zeros(knowledge.size(1), 1).cuda()
        for i in range(self.max_step):
            # prev_mem = memories[-1]
            control, q_attn = self.control(i, context, question, control)
            if self.training:
                control = control * control_mask
            controls.append(control)

            ####original implementation#####
            control_know = control.unsqueeze(1) * ((1-attn)*knowledge)

            know_attn_weight = self.attn(control_know)

            attn = torch.softmax(know_attn_weight, 1)

            read = self.read(memories, knowledge, controls)
            memory = self.write(memories, read, controls)
            if self.training:
                memory = memory * memory_mask
            memories.append(memory)
            q_attns.append(q_attn)
            attns.append(attn)

        return memories, q_attns, attns

class MACNetwork(nn.Module):
    def __init__(self, n_vocab, dim, opt, embed_hidden=300,
                 max_step=12, self_attention=False, memory_gate=False,
                 classes=5, dropout=0.15):
        super().__init__()

        self.embed = nn.Embedding(n_vocab, embed_hidden)
        self.embed.weight.requires_grad = False
        #self.lstm = nn.LSTM(embed_hidden, dim,
        #                 batch_first=True, bidirectional=True)
        self.lstm_a = RNNEncoder(embed_hidden, dim, True, dropout_p=0, n_layers=1, rnn_type="lstm")

        self.lstm_proj = nn.Linear(dim * 2, dim)
        self.mac = MACUnit(dim, max_step,
                        self_attention, memory_gate, dropout)


        #self.linear = nn.Sequential(linear(embed_hidden, dim),
        #                            nn.ELU(),
        #                            linear(dim, dim),
        #                            nn.ELU()
        #                            )

        #self.cos = nn.CosineSimilarity(eps=1e-6)
        self.classifier = nn.Sequential(linear(dim *5, dim),
                                        nn.ELU(),
                                        nn.Dropout(p=dropout),
                                        linear(dim, classes))

        self.dim = dim
        self.opt = opt
        self.classes = classes

        self.reset()

    def reset(self):
        if not self.opt.no_glove:
            self.load_embedding(self.opt.vocab_embedding)
        else:
            self.embed.weight.data.uniform_(0, 1)

        # kaiming_uniform_(self.conv[0].weight)
        # self.conv[0].bias.data.zero_()
        # kaiming_uniform_(self.conv[2].weight)
        # self.conv[2].bias.data.zero_()
        #
        # kaiming_uniform_(self.classifier[0].weight)

    def load_embedding(self, pretrained_embedding):
        self.embed.weight.data.copy_(torch.from_numpy(pretrained_embedding))



    def forward(self, question, question_len, answer_embeddings, answer_len, dropout=0.15):
        b_size = question.size(0)

        embed = self.embed(question)
        lstm_out, h = self.lstm_a(embed, torch.as_tensor(question_len))
        #print(lstm_out.size())
        #print(h.size())

        #h = self.lstm_proj(h)

        lstm_out = self.lstm_proj(lstm_out)

        # question_feature = torch.sum(embed, dim=1) / question_len.view(b_size, 1).float()
        # question_feature = self.linear(question_feature).double()


        embed_a0 = self.embed(answer_embeddings[0])
        embed_a1 = self.embed(answer_embeddings[1])
        embed_a2 = self.embed(answer_embeddings[2])
        embed_a3 = self.embed(answer_embeddings[3])
        embed_a4 = self.embed(answer_embeddings[4])

        a0_l, a1_l, a2_l, a3_l, a4_l = answer_len[:,0], answer_len[:,1], answer_len[:,2], answer_len[:,3], answer_len[:,4]

        lstm_out_a0, h0 = self.lstm_a(embed_a0, a0_l)
        lstm_out_a1, h1 = self.lstm_a(embed_a1, a1_l)
        lstm_out_a2, h2 = self.lstm_a(embed_a2, a2_l)
        lstm_out_a3, h3 = self.lstm_a(embed_a3, a3_l)
        lstm_out_a4, h4 = self.lstm_a(embed_a4, a4_l)


        lstm_out_a0 = self.lstm_proj(lstm_out_a0)

        lstm_out_a1 = self.lstm_proj(lstm_out_a1)

        lstm_out_a2 = self.lstm_proj(lstm_out_a2)

        lstm_out_a3 = self.lstm_proj(lstm_out_a3)

        lstm_out_a4 = self.lstm_proj(lstm_out_a4)

        memory0 = self.mac(lstm_out, h, lstm_out_a0)
        memory1 = self.mac(lstm_out, h, lstm_out_a1)
        memory2 = self.mac(lstm_out, h, lstm_out_a2)
        memory3 = self.mac(lstm_out, h, lstm_out_a3)
        memory4 = self.mac(lstm_out, h, lstm_out_a4)

        out = torch.cat([memory0, memory1, memory2, memory3, memory4], 1)
        out = self.classifier(out)

        #h0 = self.lstm_proj(h0)
        #h1 = self.lstm_proj(h1)
        #h2 = self.lstm_proj(h2)
        #h3 = self.lstm_proj(h3)
        #h4 = self.lstm_proj(h4)

        # answers = self.embed(answer_embeddings)
        # answers = torch.zeros(b_size, self.classes, self.opt.emb_dim).to(device)

        # # take average for each answer
        # answers[:, 0,] = torch.sum(embed_a0, dim=1) / answer_len[:, 0].view(b_size, 1).float() #check dim
        # answers[:, 1,] = torch.sum(embed_a1, dim=1) / answer_len[:, 1].view(b_size, 1).float()
        # answers[:, 2,] = torch.sum(embed_a2, dim=1) / answer_len[:, 2].view(b_size, 1).float()
        # answers[:, 3,] = torch.sum(embed_a3, dim=1) / answer_len[:, 3].view(b_size, 1).float()
        # answers[:, 4,] = torch.sum(embed_a4, dim=1) / answer_len[:, 4].view(b_size, 1).float()
        #
        # # answers = answers.view(b_size, 5, -1)
        #
        # ans_0 = self.linear(answers[:, 0,]).double()
        # ans_1 = self.linear(answers[:, 1,]).double()
        # ans_2 = self.linear(answers[:, 2,]).double()
        # ans_3 = self.linear(answers[:, 3,]).double()
        # ans_4 = self.linear(answers[:, 4,]).double()
        #
        # answers = torch.cat([ans_0, ans_1, ans_2, ans_3, ans_4], 0)

        #similarities = torch.zeros(b_size,
         #                          self.classes)  # (num_units, b_size, 5) if computing similarity b/w every memory write and answers

        #if torch.cuda.is_available():
         #   similarities = similarities.cuda()

            # calculate cosine similarity between answer feature and memory
          #  similarities[:, 0] = self.cos(h, h0)
           # similarities[:, 1] = self.cos(h, h1)
          #  similarities[:, 2] = self.cos(h, h2)
          #  similarities[:, 3] = self.cos(h, h3)
          #  similarities[:, 4] = self.cos(h, h4)

        # print("\tIn Model: input size", image.size(),
        #       "output size", out.size())

        return out


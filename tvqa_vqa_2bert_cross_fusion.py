import os

# from bidaf import BidafAttn
# from model2 import MACUnit
from model_qa import MACUnit as MAC_QA, SelfAttentionUnit, TwoLayerSelfAttention, linear
# from model.optimal_reasoning import OptimalReasoning
# from set_transformer.model import SetTransformer
from utils import save_json_pretty, load_json
from position_encoding import PositionEncoding

__author__ = "Jie Lei"

import torch
from torch import nn
import torch.nn.functional as F
# from torch.nn import TransformerEncoder, TransformerEncoderLayer

# from rnn import RNNEncoder, max_along_time
# from mlp import MLP
from transformers import BertConfig, BertForMaskedLM


class ABC(nn.Module):
    def __init__(self, opt):
        super(ABC, self).__init__()
        self.vid_flag = "imagenet" in opt.input_streams
        self.sub_flag = "sub" in opt.input_streams
        self.vcpt_flag = "vcpt" in opt.input_streams
        self.q_flag = True
        self.set_attn_flag = False
        self.bsz = opt.bsz
        self.device = opt.device
        self.classes = 5
        hidden_size_1 = opt.hsz1
        hidden_size_2 = opt.hsz2
        self.bow_size = opt.bowsz
        self.dim = hidden_size_2
        n_layers_cls = opt.n_layers_cls
        vid_feat_size = opt.vid_feat_size
        self.embedding_size = opt.embedding_size
        vocab_size = opt.vocab_size
        self.steps = opt.mac_steps
        self.clip_attn = True
        self.read_variant = 2
        dropout = 0.15
        self.aux_loss = opt.aux_loss
        self.similarity = "cos"  # 'cos' or 'inner_prod'

        if "mode" in opt and opt.mode == "valid":
            option_file_path = os.path.join("results", opt.model_dir, 'attn_config.json')
            self.model_setup = load_json(option_file_path)
            self.attn_setup = self.model_setup['attn_config']
        else:
            self.attn_setup = {"ans_to_ques": 0, "ques_to_ans": 0, "ans_to_ctx": 0, "ctx_to_ans": 0,
                               "ques_to_ctx": 1, "ctx_to_ques": 0}
            self.model_setup = {'attn_config': self.attn_setup, 'steps': self.steps, 'dropout': dropout,
                                'feat_normalize': 0, 'dim': 1, 'similarity': self.similarity,
                                'feat_normalize_bf_cls': 0, 'clip_attn': self.clip_attn,
                                'read_variant': self.read_variant,
                                'lstm_layers': 1, 'lstm_drop_out': 0}
            # if not isinstance(opt, opt.TestOptions):
            option_file_path = os.path.join(opt.results_dir, 'attn_config.json')  # not yaml file indeed
            save_json_pretty(self.model_setup, option_file_path)
        print("Active attention setup is:")
        for k, v in self.attn_setup.items():
            print("{}: {}".format(k, v))

        config = BertConfig.from_pretrained('bert-base-uncased')
        config.output_hidden_states = True


        # self.embedding = nn.Embedding(vocab_size, self.embedding_size)
        # BertConfig.output_hidden_states = True
        self.embedding = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config)
        self.vqa_embedding = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config)

        self.cross_attn = CrossEncoderLayer(d_model=self.embedding_size, heads=opt.n_heads)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_size * 2, nhead=opt.n_heads, activation='gelu')
        self.self_attn = nn.TransformerEncoder(encoder_layer, num_layers=opt.n_layers_stacked)

        self.q_gate = nn.Sigmoid()
        self.v_gate = nn.Sigmoid()
        self.s_gate = None
        in_dim = self.embedding_size * 5
        cls_dim = self.embedding_size

        if self.sub_flag:
            self.sub_embedding = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config)
            self.sub_classifier = linear(self.embedding_size * 5, 5)
            self.s_gate = nn.Sigmoid()
            # self.crosssqa_classifier = linear(self.embedding_size * 5, 5)
            in_dim = self.embedding_size * 10
            cls_dim = self.embedding_size * 2

        self.classifier = linear(self.embedding_size * 5, 5)
        self.vqa_classifier = linear(self.embedding_size * 5, 5)
        self.joint_classifier = linear(in_dim, 5)
        self.CLS = nn.Parameter(torch.zeros(1, cls_dim))

        # self.crossvqa_classifier = linear(self.embedding_size * 5, 5)

    def load_embedding(self, pretrained_embedding):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))

    def forward(self, **batch):
        bsz = batch['q'].size(0)
        CLS = self.CLS.expand(bsz, self.embedding_size * 2).unsqueeze(1).permute(1, 0, 2)  # => 1, B, D

        #        e_q = self.embedding(q)[1][-1][:, 0]
        e_a0 = self.embedding(input_ids=batch['qa0'], attention_mask=batch['qa0_mask'])[1][-4]
        e_a1 = self.embedding(input_ids=batch['qa1'], attention_mask=batch['qa1_mask'])[1][-4]
        e_a2 = self.embedding(input_ids=batch['qa2'], attention_mask=batch['qa2_mask'])[1][-4]
        e_a3 = self.embedding(input_ids=batch['qa3'], attention_mask=batch['qa3_mask'])[1][-4]
        e_a4 = self.embedding(input_ids=batch['qa4'], attention_mask=batch['qa4_mask'])[1][-4]
        e_a0_cls = e_a0[:, 0]  # 0 for CLS token
        e_a1_cls = e_a1[:, 0]
        e_a2_cls = e_a2[:, 0]
        e_a3_cls = e_a3[:, 0]
        e_a4_cls = e_a4[:, 0]

        e_vqa0 = self.vqa_embedding(input_ids=batch['vqa0'], attention_mask=batch['vqa0_mask'])[1][-4]
        e_vqa1 = self.vqa_embedding(input_ids=batch['vqa1'], attention_mask=batch['vqa1_mask'])[1][-4]
        e_vqa2 = self.vqa_embedding(input_ids=batch['vqa2'], attention_mask=batch['vqa2_mask'])[1][-4]
        e_vqa3 = self.vqa_embedding(input_ids=batch['vqa3'], attention_mask=batch['vqa3_mask'])[1][-4]
        e_vqa4 = self.vqa_embedding(input_ids=batch['vqa4'], attention_mask=batch['vqa4_mask'])[1][-4]
        e_vqa0_cls = e_vqa0[:, 0]
        e_vqa1_cls = e_vqa1[:, 0]
        e_vqa2_cls = e_vqa2[:, 0]
        e_vqa3_cls = e_vqa3[:, 0]
        e_vqa4_cls = e_vqa4[:, 0]

        QA = torch.cat([e_a0_cls, e_a1_cls, e_a2_cls, e_a3_cls, e_a4_cls], dim=-1)
        scores_txt = self.classifier(QA)
        #print(scores_txt.shape)
        VQA = torch.cat([e_vqa0_cls, e_vqa1_cls, e_vqa2_cls, e_vqa3_cls, e_vqa4_cls], dim=-1)
        scores_vid = self.vqa_classifier(VQA)

        vqa_fused0 = self.cross_attention(e_a0, e_vqa0, e_vqa0, batch['qa0_l'], batch['vqa0_l'])
        vqa_fused1 = self.cross_attention(e_a1, e_vqa1, e_vqa1, batch['qa1_l'], batch['vqa1_l'])
        vqa_fused2 = self.cross_attention(e_a2, e_vqa2, e_vqa2, batch['qa2_l'], batch['vqa2_l'])
        vqa_fused3 = self.cross_attention(e_a3, e_vqa3, e_vqa3, batch['qa3_l'], batch['vqa3_l'])
        vqa_fused4 = self.cross_attention(e_a4, e_vqa4, e_vqa4, batch['qa4_l'], batch['vqa4_l'])
        # print(vqa_fused3.shape)

        # crossVQA = torch.cat(
        #     [vqa_fused0.permute(1, 0, 2).sum(1),
        #      vqa_fused1.permute(1, 0, 2).sum(1),
        #      vqa_fused2.permute(1, 0, 2).sum(1),
        #      vqa_fused3.permute(1, 0, 2).sum(1),
        #      vqa_fused4.permute(1, 0, 2).sum(1)],
        #     dim=-1)
        #
        # scores_cross_vid = self.crossvqa_classifier(crossVQA)

        if self.sub_flag:
            e_sqa0 = self.sub_embedding(input_ids=batch['sqa0'], attention_mask=batch['sqa0_mask'])[1][-4]
            e_sqa1 = self.sub_embedding(input_ids=batch['sqa1'], attention_mask=batch['sqa1_mask'])[1][-4]
            e_sqa2 = self.sub_embedding(input_ids=batch['sqa2'], attention_mask=batch['sqa2_mask'])[1][-4]
            e_sqa3 = self.sub_embedding(input_ids=batch['sqa3'], attention_mask=batch['sqa3_mask'])[1][-4]
            e_sqa4 = self.sub_embedding(input_ids=batch['sqa4'], attention_mask=batch['sqa4_mask'])[1][-4]
            e_sqa0_cls = e_sqa0[:, 0]
            e_sqa1_cls = e_sqa1[:, 0]
            e_sqa2_cls = e_sqa2[:, 0]
            e_sqa3_cls = e_sqa3[:, 0]
            e_sqa4_cls = e_sqa4[:, 0]

            SQA = torch.cat([e_sqa0_cls, e_sqa1_cls, e_sqa2_cls, e_sqa3_cls, e_sqa4_cls], dim=-1)
            scores_sub = self.sub_classifier(SQA)

            sqa_fused0 = self.cross_attention(e_a0, e_sqa0, e_sqa0, batch['qa0_l'], batch['sqa0_l'])
            sqa_fused1 = self.cross_attention(e_a1, e_sqa1, e_sqa1, batch['qa1_l'], batch['sqa1_l'])
            sqa_fused2 = self.cross_attention(e_a2, e_sqa2, e_sqa2, batch['qa2_l'], batch['sqa2_l'])
            sqa_fused3 = self.cross_attention(e_a3, e_sqa3, e_sqa3, batch['qa3_l'], batch['sqa3_l'])
            sqa_fused4 = self.cross_attention(e_a4, e_sqa4, e_sqa4, batch['qa4_l'], batch['sqa4_l'])

            # crossSQA = torch.cat(
            #     [sqa_fused0.permute(1, 0, 2).sum(1),
            #      sqa_fused1.permute(1, 0, 2).sum(1),
            #      sqa_fused2.permute(1, 0, 2).sum(1),
            #      sqa_fused3.permute(1, 0, 2).sum(1),
            #      sqa_fused4.permute(1, 0, 2).sum(1)],
            #     dim=-1)

            # scores_cross_sub = self.crossvqa_classifier(crossSQA)

            full_blk0 = torch.cat([CLS, torch.cat([vqa_fused0, sqa_fused0], dim=-1)],
                                  dim=0)  # ==> N+1, B, 2D , +1 for CLS
            full_blk1 = torch.cat([CLS, torch.cat([vqa_fused1, sqa_fused1], dim=-1)], dim=0)
            full_blk2 = torch.cat([CLS, torch.cat([vqa_fused2, sqa_fused2], dim=-1)], dim=0)
            full_blk3 = torch.cat([CLS, torch.cat([vqa_fused3, sqa_fused3], dim=-1)], dim=0)
            full_blk4 = torch.cat([CLS, torch.cat([vqa_fused4, sqa_fused4], dim=-1)], dim=0)

            # self attention concatenated fused stream
            full_blk0 = self.self_attn(full_blk0).permute(1, 0, 2)[:, 0]
            full_blk1 = self.self_attn(full_blk1).permute(1, 0, 2)[:, 0]
            full_blk2 = self.self_attn(full_blk2).permute(1, 0, 2)[:, 0]
            full_blk3 = self.self_attn(full_blk3).permute(1, 0, 2)[:, 0]
            full_blk4 = self.self_attn(full_blk4).permute(1, 0, 2)[:, 0]

            jointSVQA = torch.cat([full_blk0, full_blk1, full_blk2, full_blk3, full_blk4], dim=-1)

            joint_scores = self.joint_classifier(jointSVQA)


        else:
            scores_sub = torch.zeros(bsz).to(self.device)
            scores_cross_sub = torch.zeros(bsz).to(self.device)

            full_blk0 = torch.cat([CLS, vqa_fused0], dim=0)  # ==> N+1, B, D , +1 for CLS
            full_blk1 = torch.cat([CLS, vqa_fused0], dim=0)
            full_blk2 = torch.cat([CLS, vqa_fused0], dim=0)
            full_blk3 = torch.cat([CLS, vqa_fused0], dim=0)
            full_blk4 = torch.cat([CLS, vqa_fused0], dim=0)

            # self attention concatenated fused stream
            full_blk0 = self.self_attn(full_blk0).permute(1, 0, 2)[:, 0]
            full_blk1 = self.self_attn(full_blk1).permute(1, 0, 2)[:, 0]
            full_blk2 = self.self_attn(full_blk2).permute(1, 0, 2)[:, 0]
            full_blk3 = self.self_attn(full_blk3).permute(1, 0, 2)[:, 0]
            full_blk4 = self.self_attn(full_blk4).permute(1, 0, 2)[:, 0]

            jointVQA = torch.cat([full_blk0, full_blk1, full_blk2, full_blk3, full_blk4], dim=-1)
            joint_scores = self.joint_classifier(jointVQA)

        scores = {
            'scores_txt': scores_txt,
            'scores_vid': scores_vid,
            'joint_scores': joint_scores,
            'scores_sub': scores_sub,
            'scores_cross_vid': torch.zeros(bsz,5).to(self.device),
            'scores_cross_sub': torch.zeros(bsz,5).to(self.device)
        }

        return scores, None

    def cross_attention(self, q, k, v, q_l, k_l):
        # k and v are same
        bsz = q.size(0)
        heads = 12
        # s_mask = torch.ones(bsz, q.size(1), k.size(1)).bool()  # [B, T1, T2]
        # # Init similarity mask using lengths
        # for i, (l_1, l_2) in enumerate(zip(q_l, k_l)):
        #     s_mask[i][:l_1, :l_2] = 0
        # s_mask = s_mask.repeat(heads, 1, 1).squeeze()
        # todo: permute axis for k, q, v before passing to cross attention
        # print(q.shape, k.shape)
        q = q.permute(1, 0, 2)  # B, N, D -> N, B, D
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)
        attn_output = self.cross_attn(q, k, v)

        return attn_output

    def GFF(self, f_0, f_1, f_2=None):
        g_0 = self.q_gate(f_0)
        g_1 = self.v_gate(f_1)

        if self.s_gate is not None and f_2 is not None:
            g_2 = self.s_gate(f_2)
            f_0_ = (1 + g_0) * f_0 + (1 - g_0) * (g_1 * f_1 + g_2 * f_2)
            f_1_ = (1 + g_1) * f_1 + (1 - g_1) * (g_0 * f_0 + g_2 * f_2)
            f_2_ = (1 + g_2) * f_2 + (1 - g_2) * (g_0 * f_0 + g_1 * f_1)
            trainable_fused_f = self.fusion_layer(torch.cat([f_0_, f_1_, f_2_], dim=-1))
            return f_0_ * f_1_ * f_2_, trainable_fused_f
        else:
            f_0_ = (1 + g_0) * f_0 + (1 - g_0) * (g_1 * f_1)
            f_1_ = (1 + g_1) * f_1 + (1 - g_1) * (g_0 * f_0)
            trainable_fused_f = self.fusion_layer(torch.cat([f_0_, f_1_], dim=-1))
            return f_0_ * f_1_, trainable_fused_f

    @staticmethod
    def get_fake_inputs(device="cuda:0"):
        bsz = 16
        q = torch.ones(bsz, 25).long().to(device)
        q_l = torch.ones(bsz).fill_(25).long().to(device)
        a = torch.ones(bsz, 5, 20).long().to(device)
        a_l = torch.ones(bsz, 5).fill_(20).long().to(device)
        a0, a1, a2, a3, a4 = [a[:, i, :] for i in range(5)]
        a0_l, a1_l, a2_l, a3_l, a4_l = [a_l[:, i] for i in range(5)]
        sub = torch.ones(bsz, 300).long().to(device)
        sub_l = torch.ones(bsz).fill_(300).long().to(device)
        vcpt = torch.ones(bsz, 300).long().to(device)
        vcpt_l = torch.ones(bsz).fill_(300).long().to(device)
        vid = torch.ones(bsz, 100, 2048).to(device)
        vid_l = torch.ones(bsz).fill_(100).long().to(device)
        return q, q_l, a0, a0_l, a1, a1_l, a2, a2_l, a3, a3_l, a4, a4_l, sub, sub_l, vcpt, vcpt_l, vid, vid_l


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class CrossEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=heads)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # print(q.shape, k.shape)
        q2 = self.norm_1(q)
        k2 = self.norm_1(k)
        v2 = self.norm_1(v)
        # print(q.shape, q2.shape)
        q2, _ = self.attn(q2, k2, v2)
        q2 = self.dropout_1(q2)

        q = q + q2
        q2 = self.norm_2(q)
        q = q + self.dropout_2(self.ff(q2))

        return q


if __name__ == '__main__':
    from config import BaseOptions
    import sys

    sys.argv[1:] = ["--input_streams" "sub"]
    opt = BaseOptions().parse()

    model = ABC(opt)
    model.to(opt.device)
    test_in = model.get_fake_inputs(device=opt.device)
    test_out = model(*test_in)
    print(test_out.size())

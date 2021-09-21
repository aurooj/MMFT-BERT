import os

from model.bidaf import BidafAttn
from model.model_qa import SelfAttentionUnit, linear
from model.optimal_reasoning import OptimalReasoning
# from set_transformer.model import SetTransformer
from utils import save_json_pretty, load_json
from position_encoding import PositionEncoding


__author__ = "Jie Lei"

import torch
from torch import nn
import torch.nn.functional as F
# from nn import TransformerEncoder, TransformerEncoderLayer

from model.rnn import RNNEncoder, max_along_time
from model.mlp import MLP
from transformers import BertConfig, BertForMaskedLM


class ABC(nn.Module):
    def __init__(self, opt):
        super(ABC, self).__init__()
        self.vid_flag = "imagenet" in opt.input_streams
        self.sub_flag = "sub" in opt.input_streams
        self.vcpt_flag = "vcpt" in opt.input_streams
        self.model_config = opt.model_config
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

        self.embedding = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config)
        self.vcpt_embedding = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config)


        if self.model_config == 1:
            bridge_hsz = 128
            out_hsz = 64
            self.txt_bridge_encoder = nn.Sequential(linear(self.embedding_size, bridge_hsz),
                                                    nn.ReLU(),
                                                    nn.LayerNorm(bridge_hsz),
                                                    linear(bridge_hsz, out_hsz),
                                                    # nn.LayerNorm(out_hsz)
                                                    )

            self.vid_bridge_encoder = nn.Sequential(linear(self.embedding_size, bridge_hsz),
                                                    nn.ReLU(),
                                                    nn.LayerNorm(bridge_hsz),
                                                    linear(bridge_hsz, out_hsz),
                                                    # nn.LayerNorm(out_hsz)
                                                    )

            # encoder_layers = TransformerEncoderLayer(out_hsz, 4, out_hsz, dropout)
            # self.transformer_encoder = TransformerEncoder(encoder_layers, 4)

            self.downsize_encoder = linear(out_hsz * 3, out_hsz)

            self.classifier = nn.Sequential(linear(out_hsz * 3 * 5, self.classes)
                                            )

            self.text_classifier = linear(self.embedding_size * 5, self.classes)

        elif self.model_config == 2:
            self.downsize_encoder = linear(self.embedding_size * 3, self.embedding_size)
            self.classifier = nn.Sequential(linear(self.embedding_size*5, self.classes)
                                            )



    def load_embedding(self, pretrained_embedding):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))

    def forward(self, q1_a0, q1_a0l, q2_a0, q2_a0l, q1_a1, q1_a1l, q2_a1, q2_a1l, q1_a2, q1_a2l, q2_a2, q2_a2l,
                q1_a3, q1_a3l, q2_a3, q2_a3l, q1_a4, q1_a4l, q2_a4, q2_a4l,
                qa0, qa0_l, qa1, qa1_l, qa2, qa2_l, qa3, qa3_l, qa4, qa4_l,
                sub, sub_l, q1vcpt, q1vcpt_l, q2vcpt, q2vcpt_l, q_vcpt, q_vcpt_l,
                v0, v0_l, v1, v1_l, v2, v2_l, v3, v3_l, v4, v4_l, vid, vid_l, vid_masks):
        # bsz = q.size(0)
        prob_matrix = None



        # e_q = self.embedding(q)[1][-2]
        e_a0 = self.embedding(qa0)[1][-1][:, 0]  # 0 for CLS token
        e_a1 = self.embedding(qa1)[1][-1][:, 0]
        e_a2 = self.embedding(qa2)[1][-1][:, 0]
        e_a3 = self.embedding(qa3)[1][-1][:, 0]
        e_a4 = self.embedding(qa4)[1][-1][:, 0]

        e_vcpt = self.vcpt_embedding(q_vcpt)[1][-1][:, 0]
        # e_v0 = self.vcpt_embedding(v0)[1][-1][:, 0]
        # e_v1 = self.vcpt_embedding(v1)[1][-1][:, 0]
        # e_v2 = self.vcpt_embedding(v2)[1][-1][:, 0]
        # e_v3 = self.vcpt_embedding(v3)[1][-1][:, 0]
        # e_v4 = self.vcpt_embedding(v4)[1][-1][:, 0]


        if self.model_config == 1:
            e_vcpt = self.vid_bridge_encoder(e_vcpt)

            txt_pred = self.text_classifier(torch.cat([e_a0, e_a1, e_a2, e_a3, e_a4], dim=-1))

            e_a0 = self.txt_bridge_encoder(e_a0)
            e_a1 = self.txt_bridge_encoder(e_a1)
            e_a2 = self.txt_bridge_encoder(e_a2)
            e_a3 = self.txt_bridge_encoder(e_a3)
            e_a4 = self.txt_bridge_encoder(e_a4)

            # e_q1a0 = self.fc(self.embedding(q1_a0)[1][-1][:, 0])  # 0 for CLS token
            # e_q1a1 = self.fc(self.embedding(q1_a1)[1][-1][:, 0])
            # e_q1a2 = self.fc(self.embedding(q1_a2)[1][-1][:, 0])
            # e_q1a3 = self.fc(self.embedding(q1_a3)[1][-1][:, 0])
            # e_q1a4 = self.fc(self.embedding(q1_a4)[1][-1][:, 0])
            #
            # e_q2a0 = self.fc(self.embedding(q2_a0)[1][-1][:, 0])  # 0 for CLS token
            # e_q2a1 = self.fc(self.embedding(q2_a1)[1][-1][:, 0])
            # e_q2a2 = self.fc(self.embedding(q2_a2)[1][-1][:, 0])
            # e_q2a3 = self.fc(self.embedding(q2_a3)[1][-1][:, 0])
            # e_q2a4 = self.fc(self.embedding(q2_a4)[1][-1][:, 0])

            # e1_vcpt = self.vcpt_embedding(q1vcpt)[1][-1][:, 0]
            # e1_vcpt = self.vid_bridge_encoder(e1_vcpt)
            #
            # e2_vcpt = self.vcpt_embedding(q2vcpt)[1][-1][:, 0]
            # e2_vcpt = self.vid_bridge_encoder(e2_vcpt)

            # e_vcpt = max_along_time(e_vcpt, vid_l)

            # VQA0 =  torch.cat([e_a0, e_q1a0, e_q2a0, e_vcpt], dim=-1)
            # VQA1 =  torch.cat([e_a1, e_q1a1, e_q2a1, e_vcpt], dim=-1)
            # VQA2 =  torch.cat([e_a2, e_q1a2, e_q2a2, e_vcpt], dim=-1)
            # VQA3 = torch.cat([e_a3, e_q1a3, e_q2a3, e_vcpt], dim=-1)
            # VQA4 = torch.cat([e_a4, e_q1a4, e_q2a4, e_vcpt], dim=-1)

            VQA0 = torch.cat([e_a0, e_vcpt], dim=-1)
            VQA1 = torch.cat([e_a1, e_vcpt], dim=-1)
            VQA2 = torch.cat([e_a2, e_vcpt], dim=-1)
            VQA3 = torch.cat([e_a3, e_vcpt], dim=-1)
            VQA4 = torch.cat([e_a4, e_vcpt], dim=-1)
            #
            # VQA0 = self.downsize_encoder(VQA0)
            # VQA1 = self.downsize_encoder(VQA1)
            # VQA2 = self.downsize_encoder(VQA2)
            # VQA3 = self.downsize_encoder(VQA3)
            # VQA4 = self.downsize_encoder(VQA4)

            VQA_total = torch.cat([VQA0, VQA1, VQA2, VQA3, VQA4], dim=-1)  # B, D*5

            # QA = torch.cat([e_a0, e_a1, e_a2, e_a3, e_a4], dim=-1)
            scores = self.classifier(VQA_total)  # B, D*5 -> B, 5

        elif self.model_config == 2:
            VQA0 = torch.cat([e_a0, e_vcpt, e_a0*e_vcpt], dim=-1)
            VQA1 = torch.cat([e_a1, e_vcpt, e_a1*e_vcpt], dim=-1)
            VQA2 = torch.cat([e_a2, e_vcpt, e_a2*e_vcpt], dim=-1)
            VQA3 = torch.cat([e_a3, e_vcpt, e_a3*e_vcpt], dim=-1)
            VQA4 = torch.cat([e_a4, e_vcpt, e_a4*e_vcpt], dim=-1)

            VQA0 = self.downsize_encoder(VQA0)
            VQA1 = self.downsize_encoder(VQA1)
            VQA2 = self.downsize_encoder(VQA2)
            VQA3 = self.downsize_encoder(VQA3)
            VQA4 = self.downsize_encoder(VQA4)


            scores = self.classifier(torch.cat([VQA0, VQA1, VQA2, VQA3, VQA4], dim=-1))
            txt_pred = None
        #
        # out = sub_out + vcpt_out + vid_out + set_attn_out  # adding zeros has no effect on backward
        return scores.squeeze(), txt_pred, None, None, None, None, None

    def stream_processor(self, lstm_mature, classifier, ctx_embed, ctx_l,
                         q_embed, q_l, qm1, qm_l1, qm2, qm_l2, a0_embed, a0_l, a1_embed, a1_l, a2_embed, a2_l, a3_embed,
                         a3_l, a4_embed, a4_l,
                         h=None, h0=None, h1=None, h2=None, h3=None, h4=None, h_ctx=None, classifier2=None,
                         vid_len_masks=None):
        bsz = q_embed.size(0)
        prob_matrix = None
        assert sum(self.attn_setup.values()) != 0, "`out` variable cannot be zero. attn_setup has all zeros."

        if self.attn_setup['ques_to_ctx']:
            if vid_len_masks is not None:
                num_blocks = vid_len_masks.size(1)
                # question-aware answer representation
                # mem_qc, clip_attn, mem_matrix = self.mac(q_embed, h, ctx_embed, vid_len_masks, num_blocks)
                mem_qc, q_attn, attn = self.mac(q_embed, h, ctx_embed, ctx_l, qm1, qm_l1, qm2, qm_l2)
                # if self.clip_attn:
                #     mem_qc = torch.sum(mem_qc * clip_attn.unsqueeze(2), dim=1)  # (b_size,512)
                # else:
                #     mem_qc = max_along_time(mem_qc, ctx_l)  # (b_sisze, dim)
            else:
                if self.sub_flag:
                    mem_qc, q_attn, attn = self.mac_txt(q_embed, h, ctx_embed, ctx_l, qm1, qm_l1, qm2, qm_l2)
                else:
                    mem_qc, q_attn, attn = self.mac_txt(q_embed, h, ctx_embed, ctx_l, qm1, qm_l1, qm2, qm_l2)
            if mem_qc[-1].size(1) == self.dim * 2:
                # mem_qc = torch.cat([mem[:, None, :] for mem in mem_qc[1:]], 1)
                # mem_qc = torch.sum(mem_qc, 1)
                mem_qc1, mem_qc2 = torch.split(mem_qc[-1], self.dim, 1)

            # answers = torch.zeros(bsz, self.classes, self.embedding_size).to(self.device)

            # take average for each answer
            ans_0 = torch.sum(a0_embed, dim=1) / a0_l.unsqueeze(1).float()  # check dim
            ans_1 = torch.sum(a1_embed, dim=1) / a1_l.unsqueeze(1).float()
            ans_2 = torch.sum(a2_embed, dim=1) / a2_l.unsqueeze(1).float()
            ans_3 = torch.sum(a3_embed, dim=1) / a3_l.unsqueeze(1).float()
            ans_4 = torch.sum(a4_embed, dim=1) / a4_l.unsqueeze(1).float()

            if self.model_setup['feat_normalize']:
                mem_qc = F.normalize(mem_qc, p=2, dim=1)
                ans_0 = F.normalize(ans_0, p=2, dim=1)
                ans_1 = F.normalize(ans_1, p=2, dim=1)
                ans_2 = F.normalize(ans_2, p=2, dim=1)
                ans_3 = F.normalize(ans_3, p=2, dim=1)
                ans_4 = F.normalize(ans_4, p=2, dim=1)

            similarities = torch.zeros(bsz,
                                       self.classes)  # (num_units, b_size, 5) if computing similarity b/w every memory write and answers

            if torch.cuda.is_available():
                similarities = similarities.cuda()

            # calculate cosine similarity between answer feature and memory
            if self.similarity == "cos":
                # if self.steps == 2:
                #     mem_qc1, mem_qc2 = mem_qc[-2], mem_qc[-1]
                #     similarities[:, 0] = self.distance(mem_qc1, ans_0) + self.distance(mem_qc2, ans_0)
                #     similarities[:, 1] = self.distance(mem_qc1, ans_1) + self.distance(mem_qc2, ans_1)
                #     similarities[:, 2] = self.distance(mem_qc1, ans_2) + self.distance(mem_qc2, ans_2)
                #     similarities[:, 3] = self.distance(mem_qc1, ans_3) + self.distance(mem_qc2, ans_3)
                #     similarities[:, 4] = self.distance(mem_qc1, ans_4) + self.distance(mem_qc2, ans_4)
                # else:
                similarities[:, 0] = self.distance(mem_qc1, ans_0) + self.distance(mem_qc2, ans_0)
                similarities[:, 1] = self.distance(mem_qc1, ans_1) + self.distance(mem_qc2, ans_1)
                similarities[:, 2] = self.distance(mem_qc1, ans_2) + self.distance(mem_qc2, ans_2)
                similarities[:, 3] = self.distance(mem_qc1, ans_3) + self.distance(mem_qc2, ans_3)
                similarities[:, 4] = self.distance(mem_qc1, ans_4) + self.distance(mem_qc2, ans_4)

            elif self.similarity == "inner_prod":
                similarities[:, 0] = torch.bmm(mem_qc.unsqueeze(1), ans_0.unsqueeze(2)).squeeze()
                similarities[:, 1] = torch.bmm(mem_qc.unsqueeze(1), ans_1.unsqueeze(2)).squeeze()
                similarities[:, 2] = torch.bmm(mem_qc.unsqueeze(1), ans_2.unsqueeze(2)).squeeze()
                similarities[:, 3] = torch.bmm(mem_qc.unsqueeze(1), ans_3.unsqueeze(2)).squeeze()
                similarities[:, 4] = torch.bmm(mem_qc.unsqueeze(1), ans_4.unsqueeze(2)).squeeze()
            #
            # # out_qc = similarities
            if self.model_setup['feat_normalize_bf_cls']:
                mem_qc = F.normalize(mem_qc, p=2, dim=1)
                h = F.normalize(h, p=2, dim=1)
                similarities = F.normalize(similarities, p=2, dim=1)

            out_qc = torch.cat([mem_qc[-1], h, similarities], 1)

            # vf_a0 = torch.cat([mem_qc[-1], ans_0], 1)
            # vf_a1 = torch.cat([mem_qc[-1], ans_1], 1)
            # vf_a2 = torch.cat([mem_qc[-1], ans_2], 1)
            # vf_a3 = torch.cat([mem_qc[-1], ans_3], 1)
            # vf_a4 = torch.cat([mem_qc[-1], ans_4], 1)
            #
            # out_qc = torch.cat([vf_a0.unsqueeze(1),
            #                     vf_a1.unsqueeze(1),
            #                     vf_a2.unsqueeze(1),
            #                     vf_a3.unsqueeze(1),
            #                     vf_a4.unsqueeze(1)], dim=1)

            out_qc = classifier(out_qc)

            if vid_len_masks is not None and self.aux_loss:
                mem_matrix.pop(0)  # remove first tensor which has all zeros
                mem_matrix = torch.stack(mem_matrix).permute(1, 0, 2, 3)

                m2a0, m2a1, m2a2, m3a3, m2a4 = self.reason.memory2answer(ans_0, ans_1, ans_2, ans_3, ans_4, bsz,
                                                                         mem_matrix)

                # prob_matrix.shape --> (bsz, mac_steps, num_blocks, #classes)
                prob_matrix = self.reason.get_probabilities(bsz, m2a0, m2a1, m2a2, m3a3, m2a4, mem_matrix)

                # out_qc = classifier2(mem_qc).unsqueeze(2)  # (B, 5, 1)
        else:
            out_qc = 0

        out = out_qc

        return out.squeeze(), prob_matrix, q_attn, attn, None

    @classmethod
    def base_encoder(cls, data, init_encoder, downsize_encoder, input_encoder):
        """ Raw data --> higher-level embedding
        Args:
            data: (N, L) for text, (N, L, D) for video
            data_mask: (N, L)
            init_encoder: word_embedding layer for text, MLP (downsize) for video
            downsize_encoder: MLP, down project to hsz
            input_encoder: multiple layer of encoder block, with residual connection, CNN, layernorm, etc
        Returns:
            encoded_data: (N, L, D)
        """
        # todo: maybe do positional encoding before passing to init_encoder
        data = downsize_encoder(init_encoder(data))
        return input_encoder(data)

    def concat_features(self, f0, f1, f2, f3, f4):
        mature_answers = torch.cat([
            f0.unsqueeze(1),
            f1.unsqueeze(1),
            f2.unsqueeze(1),
            f3.unsqueeze(1),
            f4.unsqueeze(1)
        ], dim=1)
        return mature_answers

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

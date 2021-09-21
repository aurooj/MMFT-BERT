import os

# from bidaf import BidafAttn
# from model2 import MACUnit
from model_qa import MACUnit as MAC_QA, SelfAttentionUnit, TwoLayerSelfAttention, linear
# from model.optimal_reasoning import OptimalReasoning
# from set_transformer.model import SetTransformer
from utils import save_json_pretty, load_json
# from position_encoding import PositionEncoding

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
        self.src_type_vocab = 4
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
        self.add_src_vec = opt.add_src_vec if hasattr(opt, 'add_src_vec') else 0
        self.add_src_embed = opt.add_src_embed if hasattr(opt, 'add_src_embed') else 0
        self.src_vec_dim = opt.src_vec_dim if hasattr(opt, 'src_vec_dim') else 0
        self.num_stacked_layers = opt.n_layers_stacked if hasattr(opt, 'n_layers_stacked') else 1
        self.n_heads = opt.n_heads if hasattr(opt, 'n_heads') else 12

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

        pretrained_bert = opt.pretrained_bert if hasattr(opt, 'pretrained_bert') else 1

        if pretrained_bert:
            config = BertConfig.from_pretrained('bert-base-uncased')
            config.output_hidden_states = True

            self.embedding = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config)
            if self.vcpt_flag:
                self.vqa_embedding = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config)
                self.vqa_classifier = linear(self.embedding_size * 5, 5)
        else:
            config = BertConfig()
            config.output_hidden_states = True
            self.embedding = BertForMaskedLM(config=config)
            if self.vcpt_flag:
                self.vqa_embedding = BertForMaskedLM(config=config)
                self.vqa_classifier = linear(self.embedding_size * 5, 5)

        if self.add_src_vec:
            self.fusion_dim = self.embedding_size + opt.src_vec_dim
            self.src_vec_q = nn.Parameter(torch.zeros(1, opt.src_vec_dim))
            self.src_vec_v = nn.Parameter(torch.zeros(1, opt.src_vec_dim))
            self.src_vec_s = nn.Parameter(torch.zeros(1, opt.src_vec_dim))
        elif self.add_src_embed:
            self.fusion_dim = self.embedding_size
            self.src_embedding = nn.Embedding(self.src_type_vocab, self.embedding_size)

        else:
            self.fusion_dim = self.embedding_size
        try:
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.fusion_dim, nhead=self.n_heads, activation='gelu')
        except:
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.fusion_dim, nhead=self.n_heads)
        self.joint_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        if self.num_stacked_layers == 2:
            self.joint_encoder_layer2 = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.fused_vec0 = nn.Parameter(torch.zeros(1, self.fusion_dim))
        self.q_gate = nn.Sigmoid()
        self.v_gate = nn.Sigmoid()
        self.s_gate = None
        in_dim = self.embedding_size * 2

        if self.sub_flag:
            if opt.pretrained_bert:
                print("loading pretrained sS-bert..")
                self.sub_embedding = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config)
            else:
                self.sub_embedding = BertForMaskedLM(config=config)
            self.sub_classifier = linear(self.embedding_size * 5, 5)
            self.s_gate = nn.Sigmoid()
            in_dim = self.embedding_size * 3

        self.drop = nn.Dropout(0.1)

        self.classifier = linear(self.embedding_size * 5, 5)

        self.joint_classifier = linear(self.fusion_dim * 5, 5)

        freeze_bert = opt.freeze_bert if hasattr(opt, 'freeze_bert') else 0
        if freeze_bert:
            self.freeze_bert_params()

    def load_embedding(self, pretrained_embedding):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))

    def freeze_bert_params(self):
        for param, value in self.embedding.named_parameters():
            value.requires_grad = False
        for param, value in self.vqa_embedding.named_parameters():
            value.requires_grad = False
        for param, value in self.sub_embedding.named_parameters():
            value.requires_grad = False

    def forward(self, **batch):
        bsz = batch['q'].size(0)
        fused_vector = self.fused_vec0.expand(bsz, self.fusion_dim)
        src_input = torch.tensor([0, 1, 2, 3]).expand(4, 4) if self.sub_flag \
            else torch.tensor([0, 1, 2]).expand(4, 3) #todo: make it work for any of the bsz
        if torch.cuda.is_available():
            src_input = src_input.cuda()

        prob_matrix = None

        #        e_q = self.embedding(q)[1][-1][:, 0]
        e_a0 = self.embedding(input_ids=batch['qa0'], attention_mask=batch['qa0_mask'])[1][-4][:, 0]  # 0 for CLS token
        e_a1 = self.embedding(input_ids=batch['qa1'], attention_mask=batch['qa1_mask'])[1][-4][:, 0]
        e_a2 = self.embedding(input_ids=batch['qa2'], attention_mask=batch['qa2_mask'])[1][-4][:, 0]
        e_a3 = self.embedding(input_ids=batch['qa3'], attention_mask=batch['qa3_mask'])[1][-4][:, 0]
        e_a4 = self.embedding(input_ids=batch['qa4'], attention_mask=batch['qa4_mask'])[1][-4][:, 0]

        QA = torch.cat([e_a0, e_a1, e_a2, e_a3, e_a4], dim=-1)
        scores_txt = self.classifier(QA)

        if self.vcpt_flag:
            e_vqa0 = self.vqa_embedding(input_ids=batch['vqa0'], attention_mask=batch['vqa0_mask'])[1][-4][:,
                     0]  # 0 for CLS token
            e_vqa1 = self.vqa_embedding(input_ids=batch['vqa1'], attention_mask=batch['vqa1_mask'])[1][-4][:, 0]
            e_vqa2 = self.vqa_embedding(input_ids=batch['vqa2'], attention_mask=batch['vqa2_mask'])[1][-4][:, 0]
            e_vqa3 = self.vqa_embedding(input_ids=batch['vqa3'], attention_mask=batch['vqa3_mask'])[1][-4][:, 0]
            e_vqa4 = self.vqa_embedding(input_ids=batch['vqa4'], attention_mask=batch['vqa4_mask'])[1][-4][:, 0]

            VQA = torch.cat([e_vqa0, e_vqa1, e_vqa2, e_vqa3, e_vqa4], dim=-1)
            scores_vid = self.vqa_classifier(VQA)
        else:
            scores_vid  = torch.zeros(bsz, 5).to(self.device)

        if self.sub_flag:
            e_sqa0 = self.sub_embedding(input_ids=batch['sqa0'], attention_mask=batch['sqa0_mask'])[1][-4][:, 0]
            e_sqa1 = self.sub_embedding(input_ids=batch['sqa1'], attention_mask=batch['sqa1_mask'])[1][-4][:, 0]
            e_sqa2 = self.sub_embedding(input_ids=batch['sqa2'], attention_mask=batch['sqa2_mask'])[1][-4][:, 0]
            e_sqa3 = self.sub_embedding(input_ids=batch['sqa3'], attention_mask=batch['sqa3_mask'])[1][-4][:, 0]
            e_sqa4 = self.sub_embedding(input_ids=batch['sqa4'], attention_mask=batch['sqa4_mask'])[1][-4][:, 0]

            SQA = torch.cat([e_sqa0, e_sqa1, e_sqa2, e_sqa3, e_sqa4], dim=-1)
            scores_sub = self.sub_classifier(SQA)

            if self.add_src_vec:
                src_vec_q = self.src_vec_q.expand(bsz, self.src_vec_dim)
                src_vec_v = self.src_vec_v.expand(bsz, self.src_vec_dim)
                src_vec_s = self.src_vec_s.expand(bsz, self.src_vec_dim)

                e_QA = [torch.cat([e_qa, src_vec_q], dim=-1) for e_qa in [e_a0, e_a1, e_a2, e_a3, e_a4]]
                e_a0, e_a1, e_a2, e_a3, e_a4 = e_QA[0], e_QA[1], e_QA[2], e_QA[3], e_QA[4]

                e_VQA = [torch.cat([e_vqa, src_vec_v], dim=-1) for e_vqa in [e_vqa0, e_vqa1, e_vqa2, e_vqa3, e_vqa4]]
                e_vqa0, e_vqa1, e_vqa2, e_vqa3, e_vqa4 = e_VQA[0], e_VQA[1], e_VQA[2], e_VQA[3], e_VQA[4]

                e_SQA = [torch.cat([e_sqa, src_vec_s], dim=-1) for e_sqa in [e_sqa0, e_sqa1, e_sqa2, e_sqa3, e_sqa4]]
                e_sqa0, e_sqa1, e_sqa2, e_sqa3, e_sqa4 = e_SQA[0], e_SQA[1], e_SQA[2], e_SQA[3], e_SQA[4]
            # (B, 4, D) -> (4, B, D)
            # print(e_a0.shape)
            if self.vcpt_flag and self.sub_flag:
                fuse0 = torch.cat([fused_vector.unsqueeze(1), e_a0.unsqueeze(1), e_vqa0.unsqueeze(1), e_sqa0.unsqueeze(1)],
                                  dim=1).permute(1, 0, 2)
                fuse1 = torch.cat([fused_vector.unsqueeze(1), e_a1.unsqueeze(1), e_vqa1.unsqueeze(1), e_sqa1.unsqueeze(1)],
                                  dim=1).permute(1, 0, 2)
                fuse2 = torch.cat([fused_vector.unsqueeze(1), e_a2.unsqueeze(1), e_vqa2.unsqueeze(1), e_sqa2.unsqueeze(1)],
                                  dim=1).permute(1, 0, 2)
                fuse3 = torch.cat([fused_vector.unsqueeze(1), e_a3.unsqueeze(1), e_vqa3.unsqueeze(1), e_sqa3.unsqueeze(1)],
                                  dim=1).permute(1, 0, 2)
                fuse4 = torch.cat([fused_vector.unsqueeze(1), e_a4.unsqueeze(1), e_vqa4.unsqueeze(1), e_sqa4.unsqueeze(1)],
                                  dim=1).permute(1, 0, 2)
            elif self.sub_flag and not self.vcpt_flag:
                scores_vid = torch.zeros(bsz,5).to(self.device)
                fuse0 = torch.cat([fused_vector.unsqueeze(1), e_a0.unsqueeze(1), e_sqa0.unsqueeze(1)],
                                  dim=1).permute(1, 0, 2)
                fuse1 = torch.cat([fused_vector.unsqueeze(1), e_a1.unsqueeze(1), e_sqa1.unsqueeze(1)],
                                  dim=1).permute(1, 0, 2)
                fuse2 = torch.cat([fused_vector.unsqueeze(1), e_a2.unsqueeze(1), e_sqa2.unsqueeze(1)],
                                  dim=1).permute(1, 0, 2)
                fuse3 = torch.cat([fused_vector.unsqueeze(1), e_a3.unsqueeze(1), e_sqa3.unsqueeze(1)],
                                  dim=1).permute(1, 0, 2)
                fuse4 = torch.cat([fused_vector.unsqueeze(1), e_a4.unsqueeze(1), e_sqa4.unsqueeze(1)],
                                  dim=1).permute(1, 0, 2)

            # print(fuse0.shape)
            # print(self.joint_encoder(fuse0).shape)
            if self.add_src_embed:
                # src_embed = self.src_input[None, :].expand(bsz, 4).to(self.device)

                src_embed = self.src_embedding(src_input)
                fuse0 = fuse0 + src_embed
                fuse1 = fuse1 + src_embed
                fuse2 = fuse2 + src_embed
                fuse3 = fuse3 + src_embed
                fuse4 = fuse4 + src_embed


            fuse0 = self.joint_encoder(fuse0)
            fuse1 = self.joint_encoder(fuse1)
            fuse2 = self.joint_encoder(fuse2)
            fuse3 = self.joint_encoder(fuse3)
            fuse4 = self.joint_encoder(fuse4)

            if self.num_stacked_layers == 2:
                fuse0 = (self.joint_encoder_layer2(fuse0) + fuse0).permute(1, 0, 2)[:, 0]
                fuse1 = (self.joint_encoder_layer2(fuse1) + fuse1).permute(1, 0, 2)[:, 0]
                fuse2 = (self.joint_encoder_layer2(fuse2) + fuse2).permute(1, 0, 2)[:, 0]
                fuse3 = (self.joint_encoder_layer2(fuse3) + fuse3).permute(1, 0, 2)[:, 0]
                fuse4 = (self.joint_encoder_layer2(fuse4) + fuse4).permute(1, 0, 2)[:, 0]
            else:
                fuse0 = fuse0.permute(1, 0, 2)[:, 0]
                fuse1 = fuse1.permute(1, 0, 2)[:, 0]
                fuse2 = fuse2.permute(1, 0, 2)[:, 0]
                fuse3 = fuse3.permute(1, 0, 2)[:, 0]
                fuse4 = fuse4.permute(1, 0, 2)[:, 0]

            # print(fuse0.shape)

            jointSVQA = torch.cat([fuse0, fuse1, fuse2, fuse3, fuse4], dim=-1)

            joint_scores = self.joint_classifier(jointSVQA)
            # scores = [scores_txt.squeeze(), scores_vid.squeeze(), joint_scores.squeeze(), scores_sub.squeeze()]


        else:
            print("#################{}#################".format(self.sub_flag))
            scores_sub = torch.zeros(bsz,5).to(self.device)

            if self.add_src_vec:
                src_vec_q = self.src_vec_q.expand(bsz, self.src_vec_dim)
                src_vec_v = self.src_vec_v.expand(bsz, self.src_vec_dim)

                e_QA = [torch.cat([e_qa, src_vec_q], dim=-1) for e_qa in [e_a0, e_a1, e_a2, e_a3, e_a4]]
                e_a0, e_a1, e_a2, e_a3, e_a4 = e_QA[0], e_QA[1], e_QA[2], e_QA[3], e_QA[4]

                e_VQA = [torch.cat([e_vqa, src_vec_v], dim=-1) for e_vqa in [e_vqa0, e_vqa1, e_vqa2, e_vqa3, e_vqa4]]
                e_vqa0, e_vqa1, e_vqa2, e_vqa3, e_vqa4 = e_VQA[0], e_VQA[1], e_VQA[2], e_VQA[3], e_VQA[4]

            # (B, 3, D) -> (3, B, D)
            fuse0 = torch.cat([fused_vector.unsqueeze(1), e_a0.unsqueeze(1), e_vqa0.unsqueeze(1)], dim=1).permute(1, 0,
                                                                                                                  2)
            fuse1 = torch.cat([fused_vector.unsqueeze(1), e_a1.unsqueeze(1), e_vqa1.unsqueeze(1)], dim=1).permute(1, 0,
                                                                                                                  2)
            fuse2 = torch.cat([fused_vector.unsqueeze(1), e_a2.unsqueeze(1), e_vqa2.unsqueeze(1)], dim=1).permute(1, 0,
                                                                                                                  2)
            fuse3 = torch.cat([fused_vector.unsqueeze(1), e_a3.unsqueeze(1), e_vqa3.unsqueeze(1)], dim=1).permute(1, 0,
                                                                                                                  2)
            fuse4 = torch.cat([fused_vector.unsqueeze(1), e_a4.unsqueeze(1), e_vqa4.unsqueeze(1)], dim=1).permute(1, 0, 2)

            if self.add_src_embed:

                src_embed = self.src_embedding(src_input)
                fuse0 = fuse0 + src_embed
                fuse1 = fuse1 + src_embed
                fuse2 = fuse2 + src_embed
                fuse3 = fuse3 + src_embed
                fuse4 = fuse4 + src_embed

            fuse0 = self.joint_encoder(fuse0)
            fuse1 = self.joint_encoder(fuse1)
            fuse2 = self.joint_encoder(fuse2)
            fuse3 = self.joint_encoder(fuse3)
            fuse4 = self.joint_encoder(fuse4)

            if self.num_stacked_layers == 2:
                fuse0 = (self.joint_encoder_layer2(fuse0) + fuse0).permute(1, 0, 2)[:, 0]
                fuse1 = (self.joint_encoder_layer2(fuse1) + fuse1).permute(1, 0, 2)[:, 0]
                fuse2 = (self.joint_encoder_layer2(fuse2) + fuse2).permute(1, 0, 2)[:, 0]
                fuse3 = (self.joint_encoder_layer2(fuse3) + fuse3).permute(1, 0, 2)[:, 0]
                fuse4 = (self.joint_encoder_layer2(fuse4) + fuse4).permute(1, 0, 2)[:, 0]
            else:
                fuse0 = fuse0.permute(1, 0, 2)[:, 0]
                fuse1 = fuse1.permute(1, 0, 2)[:, 0]
                fuse2 = fuse2.permute(1, 0, 2)[:, 0]
                fuse3 = fuse3.permute(1, 0, 2)[:, 0]
                fuse4 = fuse4.permute(1, 0, 2)[:, 0]


            jointVQA = torch.cat([fuse0, fuse1, fuse2, fuse3, fuse4], dim=-1)
            joint_scores = self.joint_classifier(jointVQA)
            # scores = [scores_txt.squeeze(), scores_vid.squeeze(), joint_scores.squeeze(), scores_sub]
        scores = {
            'scores_txt': scores_txt,
            'scores_vid': scores_vid,
            'joint_scores': joint_scores,
            'scores_sub': scores_sub,
            'scores_cross_vid': torch.zeros(bsz, 5).to(self.device),
            'scores_cross_sub': torch.zeros(bsz, 5).to(self.device)
        }
        return scores, None

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

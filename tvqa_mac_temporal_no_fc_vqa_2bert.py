import os

# from bidaf import BidafAttn
# from model2 import MACUnit
from model_qa import  linear
#from model.optimal_reasoning import OptimalReasoning
# from set_transformer.model import SetTransformer
from utils import save_json_pretty, load_json
#from position_encoding import PositionEncoding

__author__ = "Jie Lei"

import torch
from torch import nn
# import torch.nn.functional as F
# from torch.nn import TransformerEncoder, TransformerEncoderLayer
#
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
        self.similarity = "cos" #'cos' or 'inner_prod'

        if "mode" in opt and opt.mode == "valid":
            option_file_path = os.path.join("results", opt.model_dir, 'attn_config.json')
            self.model_setup = load_json(option_file_path)
            self.attn_setup = self.model_setup['attn_config']
        else:
            self.attn_setup = {"ans_to_ques": 0, "ques_to_ans": 0, "ans_to_ctx": 0, "ctx_to_ans": 0,
                               "ques_to_ctx": 1, "ctx_to_ques": 0}
            self.model_setup = {'attn_config':self.attn_setup, 'steps': self.steps, 'dropout': dropout,
                                'feat_normalize': 0, 'dim': 1, 'similarity':self.similarity,
                                'feat_normalize_bf_cls':0, 'clip_attn':self.clip_attn,'read_variant':self.read_variant,
                                'lstm_layers':1, 'lstm_drop_out':0}
            #if not isinstance(opt, opt.TestOptions):
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
        if self.vcpt_flag:
            self.vqa_embedding = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config)
            self.vqa_classifier = linear(self.embedding_size*5, 5)
        self.q_gate = nn.Sigmoid()
        self.v_gate = nn.Sigmoid()
        self.s_gate = None

        if self.sub_flag:
            self.sub_embedding = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config)
            self.sub_classifier = linear(self.embedding_size*5, 5)
            self.s_gate = nn.Sigmoid()

        self.drop = nn.Dropout(0.1)

        self.classifier = linear(self.embedding_size*5, 5)

        self.joint_classifier = linear(self.embedding_size*5, 5)



    def load_embedding(self, pretrained_embedding):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))

    def forward(self, **batch):
        bsz = batch['q'].size(0)
        prob_matrix = None

#        e_q = self.embedding(q)[1][-1][:, 0]
        e_a0 = self.drop(self.embedding(input_ids=batch['qa0'], attention_mask=batch['qa0_mask'])[1][-4][:,0]) # 0 for CLS token
        e_a1 = self.drop(self.embedding(input_ids=batch['qa1'], attention_mask=batch['qa1_mask'])[1][-4][:,0])
        e_a2 = self.drop(self.embedding(input_ids=batch['qa2'], attention_mask=batch['qa2_mask'])[1][-4][:,0])
        e_a3 = self.drop(self.embedding(input_ids=batch['qa3'], attention_mask=batch['qa3_mask'])[1][-4][:,0])
        e_a4 = self.drop(self.embedding(input_ids=batch['qa4'], attention_mask=batch['qa4_mask'])[1][-4][:,0])

        QA = torch.cat([e_a0, e_a1, e_a2, e_a3, e_a4], dim=-1)
        scores_txt = self.classifier(QA)

        if self.vcpt_flag:
            e_vqa0 = self.drop(self.vqa_embedding(input_ids=batch['vqa0'], attention_mask=batch['vqa0_mask'])[1][-4][:,0]) # 0 for CLS token
            e_vqa1 = self.drop(self.vqa_embedding(input_ids=batch['vqa1'], attention_mask=batch['vqa1_mask'])[1][-4][:,0])
            e_vqa2 = self.drop(self.vqa_embedding(input_ids=batch['vqa2'], attention_mask=batch['vqa2_mask'])[1][-4][:,0])
            e_vqa3 = self.drop(self.vqa_embedding(input_ids=batch['vqa3'], attention_mask=batch['vqa3_mask'])[1][-4][:,0])
            e_vqa4 = self.drop(self.vqa_embedding(input_ids=batch['vqa4'], attention_mask=batch['vqa4_mask'])[1][-4][:,0])
            VQA = torch.cat([e_vqa0, e_vqa1, e_vqa2, e_vqa3, e_vqa4], dim=-1)
            scores_vid = self.vqa_classifier(VQA)
        else:
            scores_vid = torch.zeros(bsz, 5).to(self.device)
        
        if self.sub_flag:
            e_sqa0 = self.drop(self.sub_embedding(input_ids=batch['sqa0'], attention_mask=batch['sqa0_mask'])[1][-4][:,0]) # 0 for CLS token
            e_sqa1 = self.drop(self.sub_embedding(input_ids=batch['sqa1'], attention_mask=batch['sqa1_mask'])[1][-4][:,0])
            e_sqa2 = self.drop(self.sub_embedding(input_ids=batch['sqa2'], attention_mask=batch['sqa2_mask'])[1][-4][:,0])
            e_sqa3 = self.drop(self.sub_embedding(input_ids=batch['sqa3'], attention_mask=batch['sqa3_mask'])[1][-4][:,0])
            e_sqa4 = self.drop(self.sub_embedding(input_ids=batch['sqa4'], attention_mask=batch['sqa4_mask'])[1][-4][:,0])
           
            SQA = torch.cat([e_sqa0, e_sqa1, e_sqa2, e_sqa3, e_sqa4], dim=-1)
            scores_sub = self.sub_classifier(SQA)

            # #gated fusion
            # gff0 = self.GFF(e_a0, e_vqa0, e_sqa0)
            # gff1 = self.GFF(e_a1, e_vqa1, e_sqa1)
            # gff2 = self.GFF(e_a2, e_vqa2, e_sqa2)
            # gff3 = self.GFF(e_a3, e_vqa3, e_sqa3)
            # gff4 = self.GFF(e_a4, e_vqa4, e_sqa4)

            # full_blk0 = torch.cat([e_a0 * e_vqa0 * e_sqa0, gff0], dim=-1)
            # full_blk1 = torch.cat([e_a1 * e_vqa1 * e_sqa1, gff1], dim=-1)
            # full_blk2 = torch.cat([e_a2 * e_vqa2 * e_sqa2, gff2], dim=-1)
            # full_blk3 = torch.cat([e_a3 * e_vqa3 * e_sqa3, gff3], dim=-1)
            # full_blk4 = torch.cat([e_a4 * e_vqa4 * e_sqa4, gff4], dim=-1)

            if self.sub_flag and self.vcpt_flag:

                jointSVQA = torch.cat([e_a0 * e_vqa0 * e_sqa0,
                                       e_a1 * e_vqa1 * e_sqa1,
                                       e_a2 * e_vqa2 * e_sqa2,
                                       e_a3 * e_vqa3 * e_sqa3,
                                       e_a4 * e_vqa4 * e_sqa4], dim=-1)

            elif self.sub_flag and not self.vcpt_flag:
                jointSVQA = torch.cat([e_a0 * e_sqa0,
                                       e_a1 * e_sqa1,
                                       e_a2 * e_sqa2,
                                       e_a3 * e_sqa3,
                                       e_a4 * e_sqa4], dim=-1)

            joint_scores = self.joint_classifier(jointSVQA)


        else:
            scores_sub = torch.zeros(bsz, 5).to(self.device)

            #  #gated fusion
            # gff0 = self.GFF(e_a0, e_vqa0)
            # gff1 = self.GFF(e_a1, e_vqa1)
            # gff2 = self.GFF(e_a2, e_vqa2)
            # gff3 = self.GFF(e_a3, e_vqa3)
            # gff4 = self.GFF(e_a4, e_vqa4)
            #
            # full_blk0 = torch.cat([e_a0 * e_vqa0, gff0], dim=-1)
            # full_blk1 = torch.cat([e_a1 * e_vqa1, gff1], dim=-1)
            # full_blk2 = torch.cat([e_a2 * e_vqa2, gff2], dim=-1)
            # full_blk3 = torch.cat([e_a3 * e_vqa3, gff3], dim=-1)
            # full_blk4 = torch.cat([e_a4 * e_vqa4, gff4], dim=-1)

            jointVQA = torch.cat([e_a0 * e_vqa0 ,
                                   e_a1 * e_vqa1 ,
                                   e_a2 * e_vqa2 ,
                                   e_a3 * e_vqa3 ,
                                   e_a4 * e_vqa4], dim=-1)

            joint_scores = self.joint_classifier(jointVQA)

        scores = {
            'scores_txt': scores_txt,
            'scores_vid': scores_vid,
            'joint_scores': joint_scores,
            'scores_sub': scores_sub,
            'scores_cross_vid': torch.zeros(bsz, 5).to(self.device),
            'scores_cross_sub': torch.zeros(bsz, 5).to(self.device)
        }
        return scores, None




    def GFF(self, f_0, f_1, f_2=None):
        g_0 = self.q_gate(f_0)
        g_1 = self.v_gate(f_1)

        if self.s_gate is not None and f_2 is not None:
            g_2 = self.s_gate(f_2)
            f_0_ = (1+g_0) * f_0 + (1-g_0) * (g_1*f_1 + g_2*f_2)
            f_1_ = (1+g_1) * f_1 + (1-g_1) * (g_0*f_0 + g_2*f_2)
            f_2_ = (1+g_2) * f_2 + (1-g_2) * (g_0*f_0 + g_1*f_1)
            return f_0_ * f_1_ * f_2_
        else:
            f_0_ = (1+g_0) * f_0 + (1-g_0) * (g_1*f_1)
            f_1_ = (1+g_1) * f_1 + (1-g_1) * (g_0*f_0)
            return f_0_ * f_1_



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

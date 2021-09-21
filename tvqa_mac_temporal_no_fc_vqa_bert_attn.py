import os

from bidaf import BidafAttn
from model2 import MACUnit
from model_qa import MACUnit as MAC_QA, SelfAttentionUnit, TwoLayerSelfAttention, linear
#from model.optimal_reasoning import OptimalReasoning
# from set_transformer.model import SetTransformer
from utils import save_json_pretty, load_json
from position_encoding import PositionEncoding

__author__ = "Jie Lei"

import torch
from torch import nn
import torch.nn.functional as F
# from torch.nn import TransformerEncoder, TransformerEncoderLayer

from rnn import RNNEncoder, max_along_time
from mlp import MLP
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

        # bert = BertForMaskedLM()
        # config = BertConfig.from_json_file('/data/Aisha/TVQA/Reason-VQA/fine_tune_bert/output/config.json')
        config = BertConfig.from_pretrained('bert-base-uncased')
        config.output_hidden_states = True


        # self.embedding = nn.Embedding(vocab_size, self.embedding_size)
        # BertConfig.output_hidden_states = True
        # self.embedding = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config)
        self.vqa_embedding = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config)
        # self.embedding = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path='/data/Aisha/TVQA/Reason-VQA/fine_tune_bert/output/pytorch_model.bin', config='/data/Aisha/TVQA/Reason-VQA/fine_tune_bert/output/config.json')
        # self.embedding.eval()
        # self.bidaf = BidafAttn(hidden_size_1 * 3, method="dot")  # no parameter for dot
        self.drop = nn.Dropout(0.5)
        # self.lstm_raw = RNNEncoder(self.embedding_size, hidden_size_1, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")
        # self.bridge_hsz = 500
        #
        # self.bert_word_encoding_fc = nn.Sequential(
        #     nn.LayerNorm(self.embedding_size),
        #     nn.Dropout(0.1),
        #    linear(self.embedding_size, self.bridge_hsz),
        #     nn.ReLU(True),
        #     nn.LayerNorm(self.bridge_hsz),
        # )
        #
        # self.input_embedding = nn.Sequential(
        #     nn.Dropout(0.1),
        #     linear(self.bridge_hsz,hidden_size_2),
        #     nn.ReLU(True),
        #     nn.LayerNorm(hidden_size_2),
        # )

        # self.pe = PositionEncoding(n_filters=hidden_size_2)
        self.vid_feat_size = opt.vid_feat_size
        self.visual_encoder = nn.Sequential(
            nn.LayerNorm(opt.vid_feat_size),
            nn.Dropout(0.1),
            linear(opt.vid_feat_size, self.embedding_size),
        )


        self.classifier = linear(self.embedding_size*6, 5)


        #self.lstm_decoder = linear(hidden_size_2, vocab_size)
        # self.layer_norm = nn.LayerNorm(hidden_size_2, eps=1e-6)

        # if opt.mac_version == "simple":
        #     # self.mac = SimpleMACUnit(hidden_size_2, self.steps, False, False,
        #     #                          dropout=dropout, variant=self.read_variant)
        #     self.mac = MAC_QA(hidden_size_2, self.steps,
        #                       False, False, dropout=dropout)
        # else:
        #     self.mac = MACUnit(hidden_size_2, self.steps,
        #                    False, False, dropout=dropout)
        # # self.mac = MAC_QA(hidden_size_2, self.steps,
        # #                       False, False, dropout=dropout)
        # self.mac_txt = MAC_QA(hidden_size_2, 2, self.steps,
        #                       False, False, dropout=dropout)
        # self.mac_sub = MAC_QA(hidden_size_2, self.steps,
        #                       False, False, dropout=dropout)
        #
        # self.bn_pre = nn.BatchNorm1d(self.embedding_size+1) #batch norm before lstm
        # self.bn_post = nn.BatchNorm1d(hidden_size_2) #batch norm after lstm
        #
        # if self.vid_flag:
        #     print("activate video stream")
        #
        #     self.distance = nn.CosineSimilarity(eps=1e-6)
        #     self.video_fc = nn.Sequential(
        #         nn.Dropout(0.5),
        #         nn.Linear(vid_feat_size, self.embedding_size), #if 4096, rf=4, output dim = 4096/4=1024
        #         nn.Tanh(),
        #
        #     )
        #
        #     self.lstm_mature_vid = RNNEncoder(hidden_size_1 * 2 * 5, hidden_size_2, bidirectional=True,
        #                                       dropout_p=0, n_layers=1, rnn_type="lstm")
        #     self.classifier_vid = MLP(hidden_size_2*3 + self.classes, self.classes, 500, n_layers_cls, elu=False)
        #
        #     self.classifier = MLP(hidden_size_2*5, 5, 500, n_layers_cls, elu=False)
        #
        #     self.reason = OptimalReasoning(opt, 'concat')

        # if self.sub_flag:
        #     print("activate sub stream")
        #     self.lstm_mature_sub = RNNEncoder(hidden_size_1 * 2 * 5, hidden_size_2, bidirectional=True,
        #                                       dropout_p=0, n_layers=1, rnn_type="lstm")
        #     self.classifier_sub = MLP(hidden_size_2*3 + self.classes, self.classes, 500, n_layers_cls, elu=False)
        #
        #     self.classifier_sub2 = MLP(hidden_size_2, 1, 500, n_layers_cls, elu=True)

        # if self.vcpt_flag:
        #     print("activate vcpt stream")
        #     self.lstm_vcpt = RNNEncoder(self.embedding_size+1, hidden_size_1, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")
        #     self.lstm_mature_vcpt = RNNEncoder(hidden_size_1 * 2 * 5, hidden_size_2, bidirectional=True,
        #                                        dropout_p=0, n_layers=1, rnn_type="lstm")
        #     self.classifier_vcpt = MLP(hidden_size_2*3+5, self.classes, 500, n_layers_cls, elu=False)
        #     self.lambda_txt = nn.Parameter(torch.zeros(1))
        #     self.lambda_ts = nn.Parameter(torch.rand(1))
        #     # self.setattn = TwoLayerSelfAttention(hidden_size_2+1)
        self.attn = BidafAttn(self.embedding_size, method="dot")
        #     # self.setattn2 = BidafAttn(self.embedding_size, method="dot")
        #     self.fc = nn.Sequential(linear(hidden_size_2, hidden_size_2),
        #                             nn.ReLU()
        #                             )
        #     self.fc2 = nn.Sequential(linear(hidden_size_2, hidden_size_2),
        #                             nn.ReLU()
        #                             )
        #     # self.fc3 = linear(hidden_size_2, hidden_size_2)
        #     # self.identity = linear(self.embedding_size + 1, hidden_size_2)
        #     self.layernorm = nn.LayerNorm(hidden_size_2)
        #     # self.ln = nn.LayerNorm(hidden_size_2)
        #     # self.fc_21 = linear(self.embedding_size, hidden_size_2)
        #     # self.fc_22 = linear(self.embedding_size, hidden_size_2)
        #
        #
        # if self.set_attn_flag:
        #     self.setattn = TwoLayerSelfAttention(self.embedding_size+1)
        #
        # self.distance = nn.CosineSimilarity(eps=1e-6)

    def load_embedding(self, pretrained_embedding):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))

    def forward(self, q, q_l, qm1, qm_l1, qm2, qm_l2,  vqa0, vqa0_l, a0, a0_l, vqa1, vqa1_l, a1, a1_l, vqa2, vqa2_l,
                a2, a2_l, vqa3, vqa3_l, a3, a3_l, vqa4, vqa4_l, a4, a4_l,
                sub, sub_l, vcpt, vcpt_l, vcpt_ts, vcpts_tsl, vscene, vscene_l, vscene_ts, vscene_tsl, vid, vid_l, vid_l_masks):
        bsz = q.size(0)
        prob_matrix = None

        e_q = self.embedding(q)[1][-1][:, 1:]
        visual_embed = self.visual_encoder(vid.view(bsz, -1, self.vid_feat_size))
        visual_attn_embed, _, _ = self.attn(e_q, visual_embed) #-> B, N, 768
        visual_attn_embed = visual_attn_embed.sum(1).squeeze() # sum over N to get B, 768


        # e_a0 = self.embedding(a0)[1][-1][:,0] # 0 for CLS token
        # e_a1 = self.embedding(a1)[1][-1][:,0]
        # e_a2 = self.embedding(a2)[1][-1][:,0]
        # e_a3 = self.embedding(a3)[1][-1][:,0]
        # e_a4 = self.embedding(a4)[1][-1][:,0]

        e_vqa0 = self.vqa_embedding(vqa0)[1][-1][:,0] # 0 for CLS token
        e_vqa1 = self.vqa_embedding(vqa1)[1][-1][:,0]
        e_vqa2 = self.vqa_embedding(vqa2)[1][-1][:,0]
        e_vqa3 = self.vqa_embedding(vqa3)[1][-1][:,0]
        e_vqa4 = self.vqa_embedding(vqa4)[1][-1][:,0]



        VQA = torch.cat([visual_attn_embed, e_vqa0, e_vqa1, e_vqa2, e_vqa3, e_vqa4], dim=-1)
        scores = self.classifier(VQA)


        # raw_out_q, hq = self.lstm_raw(e_q, q_l)
        # raw_out_q = self.layernorm(raw_out_q)
        # raw_out_a0, h0 = self.lstm_raw(e_a0, a0_l)
        # raw_out_a0 = self.layernorm(raw_out_a0)
        # raw_out_a1, h1 = self.lstm_raw(e_a1, a1_l)
        # raw_out_a1 = self.layernorm(raw_out_a1)
        # raw_out_a2, h2 = self.lstm_raw(e_a2, a2_l)
        # raw_out_a2 = self.layernorm(raw_out_a2)
        # raw_out_a3, h3 = self.lstm_raw(e_a3, a3_l)
        # raw_out_a3 = self.layernorm(raw_out_a3)
        # raw_out_a4, h4 = self.lstm_raw(e_a4, a4_l)
        # raw_out_a4 = self.layernorm(raw_out_a4)

        # raw_out_q = self.base_encoder(e_q, self.bert_word_encoding_fc, self.input_embedding, self.pe)
        # hq = raw_out_q[:,0]
        # raw_out_a0 = self.base_encoder(e_a0, self.bert_word_encoding_fc, self.input_embedding, self.pe)
        # raw_out_a1 = self.base_encoder(e_a1, self.bert_word_encoding_fc, self.input_embedding, self.pe)
        # raw_out_a2 = self.base_encoder(e_a2, self.bert_word_encoding_fc, self.input_embedding, self.pe)
        # raw_out_a3 = self.base_encoder(e_a3, self.bert_word_encoding_fc, self.input_embedding, self.pe)
        # raw_out_a4 = self.base_encoder(e_a4, self.bert_word_encoding_fc, self.input_embedding, self.pe)

        # pad_ = (0, 1, 0, 0, 0, 0)
        # e_q = F.pad(e_q, pad_, 'constant', -1.0)
        # e_a0 = F.pad(e_a0, pad_, 'constant', -1.0)
        # e_a1 = F.pad(e_a1, pad_, 'constant', -1.0)
        # e_a2 = F.pad(e_a2, pad_, 'constant', -1.0)
        # e_a3 = F.pad(e_a3, pad_, 'constant', -1.0)
        # e_a4 = F.pad(e_a4, pad_, 'constant', -1.0)

        # if self.vcpt_flag:
        #
        #     e_vcpt = self.drop(self.embedding(vcpt)[1][-2])
        #     e_vcpt = self.base_encoder(e_vcpt, self.bert_word_encoding_fc, self.input_embedding, self.pe)
        #
        #
        #     # e_vcpt = torch.cat([e_vcpt, self.lambda_ts * vcpt_ts[:, :, None]], dim=-1)
        #     proj11 = self.fc(e_vcpt)
        #     proj12 = self.fc2(e_vcpt)
        #     raw_out_vcpt1, raw_out_vcpt2, s = self.setattn(proj11, vcpt_l, proj12, vcpt_l)
        #     # raw_out_vcpt = self.ln(raw_out_vcpt)
        #
        #     # proj21 = self.fc_21(raw_out_vcpt)
        #     # proj22 = self.fc_22(raw_out_vcpt)
        #     # raw_out_vcpt, _ , _= self.setattn2(raw_out_vcpt1, vcpt_l, raw_out_vcpt2, vcpt_l)
        #     # raw_out_vcpt = self.fc3(raw_out_vcpt) + raw_out_vcpt1 #skip connection
        #
        #     # i_vcpt = self.identity(e_vcpt)
        #     # raw_out_vcpt = F.relu(raw_out_vcpt + i_vcpt) #skip connection
        #
        #     # e_vcpt_attended, _ = self.setattn(e_vcpt, vcpt_l, e_vcpt, vcpt_l)
        #     # e_vcpt_attended = self.fc(e_vcpt_attended)
        #     # e_vcpt = self.fc2(e_vcpt)
        #     #
        #     # raw_out_vcpt = torch.cat([e_vcpt, e_vcpt_attended], dim=1)
        #
        # if self.set_attn_flag:
        #     e_vcpt = self.embedding(vcpt)
        #
        #     e_vcpt = torch.cat([e_vcpt, vcpt_ts[:, :, None]], dim=-1)
        #     e_vcpt_attended = self.setattn(e_vcpt, vcpt_l)
        #
        #
        # q_len, a0_len, a1_len, a2_len, a3_len, a4_len = e_q.size(1), e_a0.size(1), e_a1.size(1), e_a2.size(1), e_a3.size(1), e_a4.size(1)
        #
        #
        #
        #
        # if self.sub_flag:
        #     e_sub = self.embedding(sub) #replaced sub input with vscene for now, todo: revert it back to subtitles
        #     raw_out_sub, h_sub = self.lstm_raw(e_sub, sub_l)
        #     sub_out, prob_matrix, q_attn, attn, similarities = self.stream_processor(None, self.classifier_vcpt, raw_out_sub, sub_l,
        #                                     raw_out_q, q_l, qm1, qm_l1, qm2, qm_l2, raw_out_a0, a0_l, raw_out_a1, a1_l,
        #                                     raw_out_a2, a2_l, raw_out_a3, a3_l, raw_out_a4, a4_l, hq, h0, h1, h2, h3, h4, h_sub, None)
        # else:
        #     sub_out = 0
        #
        # if self.vcpt_flag:
        #     # raw_out_vcpt, h_cpt = self.lstm_vcpt(e_vcpt, vcpt_l)
        #     h_cpt = None
        #     h0, h1, h2, h3, h4 = None, None, None, None, None
        #
        #     vcpt_out, prob_matrix, q_attn, attn, similarities = self.stream_processor(self.lstm_mature_vcpt, self.classifier_vcpt, raw_out_vcpt1, vcpt_l,
        #                                      raw_out_q, q_l, qm1, qm_l1, qm2, qm_l2, raw_out_a0, a0_l, raw_out_a1, a1_l,
        #                                      raw_out_a2, a2_l, raw_out_a3, a3_l, raw_out_a4, a4_l, hq, h0, h1, h2, h3, h4, h_cpt)
        # else:
        #     vcpt_out = 0
        #
        #
        # if self.set_attn_flag:
        #
        #     raw_out_vcpt, h_cpt = self.lstm_raw(e_vcpt_attended, vcpt_l)
        #
        #
        #     set_attn_out, prob_matrix, q_attn, attn, similarities = self.stream_processor(self.lstm_mature_vcpt, self.classifier_vcpt, raw_out_vcpt, vcpt_l,
        #                                      raw_out_q, q_l, qm1, qm_l1, qm2, qm_l2, raw_out_a0, a0_l, raw_out_a1, a1_l,
        #                                      raw_out_a2, a2_l, raw_out_a3, a3_l, raw_out_a4, a4_l, hq, h0, h1, h2, h3, h4, h_cpt)
        # else:
        #     set_attn_out = 0
        #
        # if self.vid_flag:
        #     bsz, num_blocks, f = vid.size()
        #
        #     # # flatten x in batchxunits dim
        #     # vid = vid.view(bsz * num_blocks, t, f)
        #     # out = layer(vid.float()).view(bsz, num_blocks, self.dim, t, h, w)
        #     e_vid = self.video_fc(vid)
        #     raw_out_vid, h_vid = self.lstm_raw(e_vid, vid_l)
        #     #bow = self.regressor(h_vid)
        #
        #     # raw_out_vid = raw_out_vid.view(bsz * num_blocks, self.dim)
        #
        #     # raw_out_vid, h_vid = self.lstm_raw(e_vid, vid_l)
        #     vid_out, prob_matrix, q_attn, attn, similarities = self.stream_processor(self.lstm_mature_vid, self.classifier_vid, raw_out_vid, vid_l,
        #                                     raw_out_q, q_l, qm1, qm_l1, qm2, qm_l2, raw_out_a0, a0_l, raw_out_a1, a1_l,
        #                                     raw_out_a2, a2_l, raw_out_a3, a3_l, raw_out_a4, a4_l, hq, h0, h1, h2, h3, h4,
        #                                     h_ctx=h_vid, classifier2=self.classifier, vid_len_masks=vid_l_masks)
        # else:
        #     vid_out = 0
        #
        # out = sub_out + vcpt_out + vid_out + set_attn_out  # adding zeros has no effect on backward
        return scores.squeeze(), prob_matrix, None, None, None, None, None



    def stream_processor(self, lstm_mature, classifier, ctx_embed, ctx_l,
                         q_embed, q_l, qm1, qm_l1, qm2, qm_l2, a0_embed, a0_l, a1_embed, a1_l, a2_embed, a2_l, a3_embed, a3_l, a4_embed, a4_l,
                         h=None, h0=None, h1=None, h2=None, h3=None, h4=None, h_ctx=None, classifier2=None, vid_len_masks=None):
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
            if mem_qc[-1].size(1) == self.dim*2:
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


            similarities = torch.zeros(bsz, self.classes)  # (num_units, b_size, 5) if computing similarity b/w every memory write and answers

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

                m2a0, m2a1, m2a2, m3a3, m2a4 = self.reason.memory2answer(ans_0, ans_1, ans_2, ans_3, ans_4, bsz, mem_matrix)

                # prob_matrix.shape --> (bsz, mac_steps, num_blocks, #classes)
                prob_matrix = self.reason.get_probabilities(bsz, m2a0, m2a1, m2a2, m3a3, m2a4, mem_matrix)

            #out_qc = classifier2(mem_qc).unsqueeze(2)  # (B, 5, 1)
        else:
            out_qc = 0


        out =  out_qc

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
        #todo: maybe do positional encoding before passing to init_encoder
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

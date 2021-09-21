#from model.baselines import LstmModel

__author__ = "Jie Lei"

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3"
import torch
import torch.nn as nn
from torch.nn.functional import softmax, log_softmax
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

from tvqa_mac_temporal_no_fc_vqa_bert import ABC
from tvqa_dataset_vqa_bert import TVQADataset, pad_collate, preprocess_inputs
from config import BaseOptions
from utils import clip_gradients
#from optimization import AdamW


#class for entropy loss
class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        # le is entropy loss
        le = - torch.mean(torch.mul(softmax(x), log_softmax(x)))
        return le

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, x):
        #x is an nxn score matrix with dimensions: B, N, N
        x = torch.nn.functional.softmax(x, dim=2)
        x.data.masked_fill_(x.data != x.data, 0) # remove nan from softmax on -inf
        column_sum = x.sum(1) - 1 #row-wise sum
        column_sum.data.masked_fill(column_sum.data <= 0, 0) #if sum <=1, loss is 0
        loss = - torch.mean(column_sum.sum(1))
        return loss

def l2_regularization(model, lambda_=5e-4):
    l2_reg  =0
    for param, value in model.named_parameters():
        if param.endswith('weight'):
            l2_reg += lambda_ * torch.norm(value, 2)
    all_params = torch.cat([x.view(-1) for x in model.mac_txt.attn.parameters()])

    l1_reg = lambda_ * torch.norm(all_params, 2)

    return l1_reg

def train(opt, dset, model, criterion, optimizer, epoch, previous_best_acc, criterion2):
    dset.set_mode("train")
    model.train()
    train_loader = DataLoader(dset, batch_size=opt.bsz, shuffle=True, collate_fn=pad_collate, drop_last=True, num_workers=16)

    train_loss = []
    valid_acc_log = ["batch_idx\tacc"]
    train_corrects = []
    txt_corrects = []

    torch.set_grad_enabled(True)
    # reason_graph = ReasoningGraph()
    for batch_idx, batch in tqdm(enumerate(train_loader)):
        model_inputs, targets, qids = preprocess_inputs(batch, opt.max_sub_l, opt.max_vcpt_l, opt.max_vid_l,
                                                        device=opt.device)
        if opt.train_baseline:
            outputs = model(*model_inputs)
        else:
            outputs, txt_outputs,_,_,_, bow, s = model(*model_inputs)

        if opt.model_config == 1:
            aux_loss = criterion2(txt_outputs, targets)
            aux_preds = txt_outputs.data.squeeze().max(1)[1]
            txt_corrects += aux_preds.eq(targets.data).cpu().numpy().tolist()
        else:
            aux_loss = 0
            txt_corrects = [0]
        loss = criterion(outputs, targets) #+ l1_regularization(model, opt.lambda_) #l1 reg on vcpts attn#+ criterion_mse(bow, bow_targets)
        total_loss = opt.wt1 * loss + opt.wt2 * aux_loss
        print(total_loss.item()/opt.bsz)
        optimizer.zero_grad()
        total_loss.backward()

        if opt.clip:
            clip_gradients(model, opt.clip)
        optimizer.step()

        # measure accuracy and record loss
        train_loss.append(total_loss.item())

        #simply take classifier's output
        pred_ids = outputs.data.squeeze().max(1)[1]

        train_corrects += pred_ids.eq(targets.data).cpu().numpy().tolist()

        if batch_idx % opt.log_train_freq == 0:
            niter = epoch * len(train_loader) + batch_idx

            train_acc_ = sum(train_corrects) / float(len(train_corrects))

            train_loss_ = sum(train_loss) / float(len(train_corrects))

            txt_acc_ = sum(txt_corrects) / float(len(txt_corrects))

            print(" Train Epoch %d loss %.4f acc %.4f txt_acc %.4f"
                      % (epoch, train_loss_, train_acc_, txt_acc_))

        if batch_idx % opt.log_freq == 0:
            niter = epoch * len(train_loader) + batch_idx

            train_acc = sum(train_corrects) / float(len(train_corrects))

            train_loss = sum(train_loss) / float(len(train_corrects))

            txt_acc = sum(txt_corrects) / float(len(txt_corrects))

            opt.writer.add_scalar("Train/Acc", train_acc, niter)
            opt.writer.add_scalar("Train/Loss", train_loss, niter)

            # Test
            valid_acc, valid_loss, valid_txt_acc = validate(opt, dset, model, mode="valid")
            opt.writer.add_scalar("Valid/Loss", valid_loss, niter)

            valid_log_str = "%02d\t%.4f" % (batch_idx, valid_acc)
            valid_acc_log.append(valid_log_str)
            if valid_acc > previous_best_acc:
                previous_best_acc = valid_acc
                torch.save(model.state_dict(), os.path.join(opt.results_dir, "best_valid.pth"))

            print(" Train Epoch %d loss %.4f acc %.4f txt_acc %.4f Val loss %.4f acc %.4f txt_acc %.4f"
                      % (epoch, train_loss, train_acc, txt_acc, valid_loss, valid_acc, valid_txt_acc))

            # reset to train
            torch.set_grad_enabled(True)
            model.train()
            dset.set_mode("train")
            train_corrects = []
            train_loss = []

        if opt.debug:
            break

    # additional log
    with open(os.path.join(opt.results_dir, "valid_acc.log"), "a") as f:
        f.write("\n".join(valid_acc_log) + "\n")

    return previous_best_acc, valid_loss


def validate(opt, dset, model, mode="valid"):
    dset.set_mode(mode)
    torch.set_grad_enabled(False)
    model.eval()
    valid_loader = DataLoader(dset, batch_size=opt.test_bsz, shuffle=False, collate_fn=pad_collate, drop_last=True, num_workers=16)

    valid_qids = []
    valid_loss = []
    valid_corrects = []
    txt_corrects = []
    for _, batch in enumerate(valid_loader):
        model_inputs, targets, qids = preprocess_inputs(batch, opt.max_sub_l, opt.max_vcpt_l, opt.max_vid_l,
                                                         device=opt.device)
        if opt.train_baseline:
            outputs = model(*model_inputs)
        else:
            outputs, txt_outputs, _, _, _, bow, s = model(*model_inputs)
        # optimal_probs = torch.zeros(getattr(batch, 'batch_len'), 5).to(opt.device)  # answer choices = k

        if opt.model_config == 1:
           aux_loss = criterion2(txt_outputs, targets)
           aux_preds = txt_outputs.data.squeeze().max(1)[1]
           txt_corrects += aux_preds.eq(targets.data).cpu().numpy().tolist()
        else:
           aux_loss = 0
           txt_corrects = [0]

        loss = criterion(outputs, targets)#+ l1_regularization(model, opt.lambda_) #l1 reg on vcpts attn #+ criterion_mse(bow, bow_targets)
        total_loss = opt.wt1 * loss + opt.wt2 * aux_loss
        # measure accuracy and record loss
        valid_qids += [int(x) for x in qids]
        valid_loss.append(total_loss.item())

        #simply take classifier's output
        pred_ids = outputs.data.squeeze().max(1)[1]

        valid_corrects += pred_ids.eq(targets.data).cpu().numpy().tolist()

        if opt.debug:
            break

    valid_acc = sum(valid_corrects) / float(len(valid_corrects))
    valid_txt_acc = sum(txt_corrects) / float(len(txt_corrects))

    valid_loss = sum(valid_loss) / float(len(valid_corrects))
    return valid_acc, valid_loss, valid_txt_acc


def freeze_param_by_key(model, key):
    for param, value in model.named_parameters():
        if key in param:
            value.requires_grad = False
    return model



if __name__ == "__main__":
    opt = BaseOptions().parse()
    # opt.random_seed = random.randint(1, 10000)
    # print("Random Seed: ", opt.random_seed)
    # torch.manual_seed(opt.random_seed)
    torch.manual_seed(7147) #previous was: 2018
    train_model_further = False

    writer = SummaryWriter(opt.results_dir)
    opt.writer = writer

    dset = TVQADataset(opt)
    opt.vocab_size = len(dset.word2idx)
    if opt.train_baseline:
        model = LstmModel(opt)
    else:
        model = ABC(opt)
    # print(model)
    if not opt.no_glove:
        print("loading Glove vocab..")
        model.load_embedding(dset.vocab_embedding)
    # else:
    #     print("loading bert text vocab..")
    #     model.load_embedding(dset.bert_embedding)  # fast text

    model.to(opt.device)
    if train_model_further:
        print("loading pre-trained model to train further..")
        model_path = os.path.join("results", "results_2020_03_08_17_16_32", "best_valid.pth")
        model.load_state_dict(torch.load(model_path))
        print("successfully loaded pre-trained model..")
    cudnn.benchmark = True
    # print("loss is cross-entropy..")
    criterion = nn.CrossEntropyLoss(reduction='sum').to(opt.device)
    criterion2 = nn.CrossEntropyLoss(reduction='sum').to(opt.device)
    # criterion2 = CustomLoss().to(opt.device)


    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=opt.lr, weight_decay=opt.wd)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
    #                             lr = opt.lr, weight_decay=opt.wd)

    print(optimizer)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1)

    #multi-gpu support
    if torch.cuda.device_count() > 1 and opt.multi_gpu:
        print("Let's use {} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    best_acc = 0.
    early_stopping_cnt = 0
    early_stopping_flag = False
    for epoch in range(opt.n_epoch):
        if not early_stopping_flag:
            # train for one epoch, valid per n batches, save the log and the best model

            cur_acc, valid_loss = train(opt, dset, model, criterion, optimizer, epoch, best_acc, criterion2)
            # scheduler.step(cur_acc)

            # remember best acc
            is_best = cur_acc > best_acc
            best_acc = max(cur_acc, best_acc)
            if not is_best:
                early_stopping_cnt += 1
                if early_stopping_cnt >= opt.max_es_cnt:
                    early_stopping_flag = True
        else:
            print("early stop flag with valid acc %.4f" % best_acc)
            opt.writer.export_scalars_to_json(os.path.join(opt.results_dir, "all_scalars.json"))
            opt.lr = opt.lr * 0.1 #comment out if don't want to decay learning rate
            # if opt.lr > 1e-4:
            #     opt.lr = opt.lr - 1e-4
            # else:
            #     opt.lr = opt.lr * 0.1
            # criterion = torch.optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
            # opt.margin = opt.margin + 0.1
            model.load_state_dict(torch.load(os.path.join(opt.results_dir, "best_valid.pth")))
            # model = freeze_param_by_key(model, 'lstm_raw') #freezing lstm
            # criterion = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, weight_decay=opt.wd)

            early_stopping_flag = False
            early_stopping_cnt = 0
            epoch = epoch - 1  # go back to previous epoch
            # opt.writer.close()
            # break  # early stop break

        if opt.debug:
            break


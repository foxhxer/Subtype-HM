import torch
from Subtype_HM import Network
from metric import vaild_survival
from torch.utils.data import Dataset
import torch.nn as nn
from torch.optim import lr_scheduler
import numpy as np
import argparse
import random
from MCEA import Align_Loss
from dataloader import load_data_fea
from utils_1 import lifeline_analysis
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
# from scipy.optimize import linear_sum_assignment
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["OMP_NUM_THREADS"] = "2"  # 控制线程数为4

# MNIST-USPS
# BDGP
# CCV
# Fashion
# Caltech-2V
# Caltech-3V
# Caltech-4V
# Caltech-5V
Dataname = "UCEC"
cancer_dict = {'BRCA': 5, 'BLCA': 5, 'KIRC': 4,
               'LUAD': 3, 'PAAD': 2, 'SKCM': 4,
               'STAD': 3, 'UCEC': 4, 'UVM': 4, 'GBM': 2}
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--temperature_l", default=0.5)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=200)
parser.add_argument("--con_epochs", default=50)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)
parser.add_argument('--h_layer_num', default=3, type=int)
parser.add_argument('--view', default=4, type=int)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The code has been optimized.
# The seed was fixed for the performance reproduction, which was higher than the values shown in the paper.

if args.dataset == "KIRC":
    args.learning_rate = 0.00001
    args.view = 4
    args.batch_size = 256
    args.h_layer_num = 4
    args.feature_dim = 128
    args.mse_epochs = 500
    args.con_epochs = 500
    seed = 10
if args.dataset == "UCEC":
    args.learning_rate = 0.00001
    args.view = 4
    args.batch_size = 256
    args.h_layer_num = 5
    args.feature_dim = 128
    args.mse_epochs = 500
    args.con_epochs = 2000
    seed = 10
if args.dataset == "PAAD":
    args.learning_rate = 0.00001
    args.view = 4
    args.batch_size = 256
    args.h_layer_num = 4
    args.feature_dim = 128
    args.mse_epochs = 500
    args.con_epochs = 1000
    seed = 10
if args.dataset == "STAD":
    args.learning_rate = 0.00001
    args.view = 4
    args.batch_size = 512
    args.h_layer_num = 9
    args.feature_dim = 128
    args.mse_epochs = 500
    args.con_epochs = 1500
    seed = 10
if args.dataset == "SKCM":
    args.learning_rate = 0.00001
    args.view = 4
    args.h_layer_num = 5
    args.feature_dim = 128
    args.mse_epochs = 1000
    args.con_epochs = 1000
    seed = 10

if args.dataset == "LUAD":
    args.learning_rate = 0.00001
    args.view = 4
    args.feature_dim = 256
    args.h_layer_num = 14
    args.feature_dim = 128
    args.mse_epochs = 200
    args.con_epochs = 4000
    seed = 10

if args.dataset == "UVM":
    args.learning_rate = 0.00001
    args.view = 4
    args.feature_dim = 256
    args.h_layer_num = 3
    args.feature_dim = 128
    args.mse_epochs = 500
    args.con_epochs = 1700
    seed = 10

if args.dataset == "BRCA":
    args.learning_rate = 0.00001
    args.view = 4
    args.h_layer_num = 3
    args.feature_dim = 256
    args.mse_epochs = 1000  # 1200
    args.con_epochs = 1300   # 100
    seed = 10

if args.dataset == "BLCA":
    args.learning_rate = 0.00001
    args.view = 4
    args.batch_size = 512
    args.h_layer_num = 9
    args.feature_dim = 128
    args.mse_epochs = 1000
    args.con_epochs = 300
    seed = 10
if args.dataset == "GBM":
    args.learning_rate = 0.00001
    args.view = 3
    args.batch_size = 256
    args.h_layer_num = 8
    args.feature_dim = 128
    args.mse_epochs = 500
    args.con_epochs = 200
    seed = 10


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


# setup_seed(seed)


# dataset, dims, view, data_size, class_num = load_data(args.dataset)
dataset, dims, data_size = load_data_fea(args.dataset,"/home/foxhx/Dataset/Benchmark/")
print("data_size:",data_size)
view = args.view

class_num = cancer_dict[args.dataset]
data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers= 1
    )



def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, _, xrs, _ = model.forward_pre(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))



def contrastive_train(epoch):
    model.train()
    tot_loss = 0.
    mes = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        # 定义标签：X1 -> 0, X2 -> 1, X3 -> 2, X4 -> 3
        labels = torch.cat([
            torch.zeros(xs[0].shape[0], dtype=torch.long),  # 标签 0
            torch.ones(xs[0].shape[0], dtype=torch.long),  # 标签 1
            torch.full((xs[0].shape[0],), 2, dtype=torch.long),  # 标签 2
            torch.full((xs[0].shape[0],), 3, dtype=torch.long)  # 标签 3
        ]).to(device)
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        hs, qs, xrs, zs, logits_m = model(xs)
        loss_list = []
        for v in range(view):
            for w in range(v+1, view):
                # loss_list.append(criterion.forward_feature(hs[v], hs[w]))
                loss_list.append(criterion.forward_label(qs[v], qs[w]))
            loss_list.append(criterion.Entropy(qs[v]))
            loss_list.append(mes(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss_d = criterion_d(logits_m, labels)
        loss = loss + loss_d
        loss.backward()
        optimizer.step()
        # scheduler.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))




accs = []
nmis = []
purs = []
if not os.path.exists('./models'):
    os.makedirs('./models')
T = 1
cnt_p = 0
log10p_p = 0
p_p = 0
for i in range(T):
    print(args.dataset)
    print("ROUND:{}".format(i+1))
    setup_seed(seed)
    model = Network(view, dims, args.feature_dim, class_num, args.h_layer_num, device)
    # print(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.con_epochs, eta_min=0.000000001)
    criterion = Align_Loss(args.batch_size, class_num, args.temperature_l, device).to(device)
    criterion_d = nn.CrossEntropyLoss()  # 交叉熵损失
    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))

    epoch = 1

    while epoch <= args.mse_epochs:
        pretrain(epoch)
        epoch += 1
    log10p, cnt,survival_results,p = vaild_survival(args.dataset, model, device, dataset, view, data_size, class_num, dataset.survival)
    while epoch <= args.mse_epochs+args.con_epochs:
        contrastive_train(epoch)
        # if epoch == args.mse_epochs + args.con_epochs:
        #     log10p,cnt = vaild_survival(args.dataset, model, device, dataset, view, data_size, class_num,dataset.survival)
        if epoch % 20 == 0:
            log10p, cnt,survival_results,p = vaild_survival(args.dataset, model, device, dataset, view, data_size, class_num, dataset.survival)
            # df = survival_results
            if cnt > cnt_p:
                cnt_p = cnt
                log10p_p = log10p
                p_p = p
                df = survival_results
                best_model = model
            elif cnt == cnt_p:
                if log10p > log10p_p:
                    log10p_p = log10p
                    p_p = p
                    df = survival_results
                    best_model = model

        epoch += 1

    state = model.state_dict()
    torch.save(state, './models/' + args.dataset + '.pth')
    # df = survival_results
    # cnt_p = cnt
    # log10p_p = log10p
    print(f'cnt_max: {cnt_p}, log10p_max: {log10p_p}')
    lifeline_analysis(df, args.dataset, p_p)
    print('Saving..')
    epoch += 1

print('--------args----------')
for k in list(vars(args).keys()):
    print('%s: %s' % (k, vars(args)[k]))

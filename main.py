import os
import myparser
import torch
import shutil
import numpy as np
import torch.nn as nn
import prepare_net_data
import torch.optim as optim
import torch.nn.functional as F
from datetime import *
from model import DP2Net
from sklearn import metrics
from utils import TensorboardWriter
from torch.optim import lr_scheduler
from polyvore_u import PloyvoreDataset

args = myparser.parse_args()


def save_checkpoint(args, state, auc_is_best):
    directory = os.path.join(args.save_dir, args.dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if auc_is_best:
        shutil.copyfile(filename, directory + '/model_best_auc_u.pth.tar')


def ndcg_score(y_score, y_label):
    y_score = np.array(y_score)
    y_label = np.array(y_label)
    order = np.argsort(np.negative(y_score))
    p_label = np.take(y_label, order)
    i_label = np.sort(y_label)[::-1]
    p_gain = 2 ** p_label - 1
    i_gain = 2 ** i_label - 1
    discounts = np.log2(np.maximum(np.arange(len(y_label)) + 1, 2.0))
    dcg_score = (p_gain / discounts).cumsum()
    idcg_score = (i_gain / discounts).cumsum()
    return dcg_score / idcg_score


def mean_ndcg_score(u_scores, u_labels):
    num_users = len(u_scores)
    n_samples = [len(scores) for scores in u_scores]
    max_sample = max(n_samples)
    count = np.zeros(max_sample)
    mean_ndcg = np.zeros(num_users)
    avg_ndcg = np.zeros(max_sample)
    for u in range(num_users):
        ndcg = ndcg_score(u_scores[u], u_labels[u])
        avg_ndcg[: n_samples[u]] += ndcg
        count[: n_samples[u]] += 1
        mean_ndcg[u] = ndcg.mean()
    return mean_ndcg, avg_ndcg / count


def train(args, model, device, train_loader, optimizer, epoch, visual_feat, L, outfit_map, corpus_feature):
    model.train()
    loss_history = []
    auc_history = []
    
    for batch_idx, (uid, pos_outfit_id, neg_outfit_id) in enumerate(train_loader):
        bsz = uid.shape[0]
        uid, pos_outfit_id, neg_outfit_id = uid.to(device), pos_outfit_id.to(device), neg_outfit_id.to(device)
        l_reg, g_ug, g_us_n, g_us_p = model(uid, pos_outfit_id, neg_outfit_id, visual_feat, L, outfit_map, corpus_feature, device, 'train')

        maxi = nn.LogSigmoid()(g_us_p - g_us_n)
        l_us = -1 * torch.mean(maxi)
        maxi = nn.LogSigmoid()(g_ug - g_us_n)
        l_ug = -1 * torch.mean(maxi)

        g_us_p, g_us_n = torch.sigmoid(g_us_p), torch.sigmoid(g_us_n)
        r_label = torch.full((bsz, 1), 1, dtype=torch.float, device=device)
        f_label = torch.full((bsz, 1), 0, dtype=torch.float, device=device)
        labels = torch.cat([r_label, f_label], 1)

        pred = torch.cat([g_us_p, g_us_n], 1)
        pred = pred.cpu().data.numpy().reshape(-1)
        labels = labels.cpu().data.numpy().reshape(-1)
        auc = metrics.roc_auc_score(labels, pred)

        loss = torch.mean(args.lamda4 * l_reg) + args.lamda2 * l_us + args.lamda3 * l_ug

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.data)
        auc_history.append(auc)

        if batch_idx % args.log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAuc:{:.4F}\tTime: {}'
                .format(epoch, (batch_idx + 1), len(train_loader), 100. * (batch_idx + 1) / len(train_loader),
                        loss.item(), auc, datetime.now()))

    loss_ = torch.stack(loss_history).mean()
    writer.update_loss(loss_, epoch, 'train_loss')
    auc_ = np.mean(auc_history)
    writer.update_loss(auc_, epoch, 'train_auc')


def test_auc(args, n_users, model, device, test_loader, visual_feat, L, outfit_map, corpus_feature):
    model.eval()
    outputs = []
    targets = []
    u_id = []
    with torch.no_grad():
        for uid, outfit_id, label in test_loader:
            bsz = uid.shape[0]
            pos_outfit_id = None
            uid, outfit_id, label = uid.to(device), outfit_id.to(device), label.to(device)
            _, g_us = model(uid, pos_outfit_id, outfit_id, visual_feat, L, outfit_map, corpus_feature, device, 'test')

            g_us = torch.sigmoid(g_us)
            num_users = n_users
            label = label.to(torch.float).unsqueeze(-1)

            outputs.append(g_us.view(-1))
            targets.append(label.view(-1))
            u_id.append(uid.view(-1))
            
        outputs = torch.cat(outputs).cpu().data.numpy()
        targets = torch.cat(targets).cpu().data.numpy()
        u_id = torch.cat(u_id).cpu().data.numpy()
        auc = metrics.roc_auc_score(targets, outputs)

        scores = [[] for u in range(num_users)]
        labels = [[] for u in range(num_users)]
        for n, u in enumerate(u_id):
            scores[u].append(outputs[n].item())
            labels[u].append(targets[n])
        scores, labels = np.array(scores, dtype=object), np.array(labels, dtype=object)
        mean_ndcg, avg_ndcg = mean_ndcg_score(scores, labels)
        ndcg = mean_ndcg.mean()

    writer.update_loss(auc, epoch, 'val_auc')
    writer.update_loss(ndcg, epoch, 'ndcg')
    print('Test Epoch: Auc:{:.4f}\tNDCG:{:.4f}'.format(auc, ndcg))
    return ndcg, auc




if __name__ == '__main__':
    device = torch.device("cuda:{}".format(args.gpu_id))
    torch.manual_seed(args.seed)

    data_generator = prepare_net_data.Data(path=args.data_path + args.dataset)
    n_users = data_generator.n_users
    n_outfits = data_generator.n_outfits
    n_items = data_generator.n_items
    visual_feat = data_generator.visual_feat
    L = data_generator.L
    user_map = data_generator.user_map
    outfit_map = data_generator.outfit_map
    corpus_feature = data_generator.corpus_feature

    train_loader = torch.utils.data.DataLoader(
        PloyvoreDataset(args, 'train'),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True)
    test_auc_loader = torch.utils.data.DataLoader(
        PloyvoreDataset(args, 'test'),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False)

    wirte_dir = os.path.join(args.save_dir, args.dataset)
    writer = TensorboardWriter(wirte_dir)
    
    model = DP2Net(args=args, n_users=n_users, n_outfits=n_outfits, n_items=n_items)
    model.to(device)
    # directory = os.path.join(args.save_dir, args.dataset)
    # filename = os.path.join(directory, 'model_best_auc_u.pth.tar')
    # state_dict = torch.load(filename)
    # model.load_state_dict(state_dict['state_dict'])  
    
    visual_feat = visual_feat.to(device)
    L = L.to(device)
    outfit_map = torch.Tensor(outfit_map).to(device)
    corpus_feature = corpus_feature.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_auc = 0
    best_ndcg = 0
    for epoch in range(1, args.epoch + 1):
        train(args, model, device, train_loader, optimizer, epoch, visual_feat, L, outfit_map, corpus_feature)
        ndcg, auc = test_auc(args, n_users, model, device, test_auc_loader, visual_feat, L, outfit_map, corpus_feature)
        
        auc_is_best = auc > best_auc
        if auc_is_best:
            best_auc = auc
            best_ndcg = ndcg
            stopping_step = 0
            print("best_ndcg:{:.4f}\nbest_auc_u:{:.4f}".format(best_ndcg, best_auc))
        elif stopping_step < args.early_stopping_patience:
            stopping_step += 1
            print("best_ndcg:{:.4f}\nbest_auc_u:{:.4f}".format(best_ndcg, best_auc))
            print('#####Early stopping steps: %d #####' % stopping_step)
        else:
            print("best_ndcg:{:.4f}\nbest_auc_u:{:.4f}".format(best_ndcg, best_auc))
            print('#####Early stop! #####')
            break
        
        save_checkpoint(args,
                        {'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'ndcg': ndcg,
                         'auc_u': auc
                         }, auc_is_best)

        scheduler.step()
    
    

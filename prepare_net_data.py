import os
import torch
import numpy as np
import scipy.sparse as sp



class Data(object):
    def __init__(self, path):
        self.path = path

        self.item_feature_file = os.path.join(self.path, 'item_feature.npy')
        self.visual_feat = np.load(self.item_feature_file)
        self.visual_feat = [[float(x) for x in row] for row in self.visual_feat]
        self.visual_feat = torch.Tensor(self.visual_feat)
        
        self.n_items = len(self.visual_feat)

        self.outfit_map_file = os.path.join(self.path, 'outfit.npy')
        self.outfit_map = np.load(self.outfit_map_file).tolist()

        self.n_outfits = len(self.outfit_map)

        self.user_map_file = os.path.join(self.path, 'user.npy')
        self.user_map = np.load(self.user_map_file, allow_pickle=True)

        self.n_users = len(self.user_map)

        self.user_map_train_file = os.path.join(self.path, 'user_train.npy')
        self.user_map_train = np.load(self.user_map_train_file, allow_pickle=True)
        
        M = sp.dok_matrix((self.n_users, self.n_outfits), dtype=np.float32) # user-outfit interaction matrix
        user_id_list = self.user_map_train.tolist()
        for i in range(self.n_users):
            o_list = user_id_list[i]
            for j in range(len(o_list)):
                oid = o_list[j]
                M[i, oid] = 1
        adj_mat = sp.dok_matrix((self.n_users + self.n_outfits, self.n_users + self.n_outfits), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        M = M.tolil()
        adj_mat[:self.n_users, self.n_users:] = M
        adj_mat[self.n_users:, :self.n_users] = M.T
        adj_mat = adj_mat.todok()
        
        def mean_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()
        
        norm_adj_mat = mean_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        norm_adj_mat = norm_adj_mat.tocsr()
        
        coo = norm_adj_mat.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        self.L = torch.sparse.FloatTensor(i, v, coo.shape)

        # feature of fashion corpus
        self.corpus_path = os.path.join(self.path, 'corpus_features.npy')
        self.corpus_feature = np.load(self.corpus_path)
        self.corpus_feature = [[float(x) for x in row] for row in self.corpus_feature]
        self.corpus_feature = torch.Tensor(self.corpus_feature)






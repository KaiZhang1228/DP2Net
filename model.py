import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList



class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        mlp = []
        linear = nn.Linear(self.input_size, self.input_size // 2)
        nn.init.xavier_uniform_(linear.weight)
        nn.init.constant_(linear.bias, 0)
        dropout = nn.Dropout(p=0.2)
        mlp.append(dropout)
        mlp.append(linear)
        mlp.append(nn.LeakyReLU())
        linear = nn.Linear(self.input_size // 2, self.output_size)
        nn.init.xavier_uniform_(linear.weight)
        nn.init.constant_(linear.bias, 0)
        mlp.append(linear)
        self.mlp = nn.Sequential(*mlp)

    def forward(self, input):
        out = self.mlp(input)
        return out


class I2I(nn.Module):
    def __init__(self, embed_size, n_items, args):
        super(I2I, self).__init__()
        self.embed_size = embed_size
        self.n_items = n_items
        self.relu = nn.LeakyReLU()
        self.drop = nn.Dropout(0.2)
        self.fc1 = nn.Linear(self.embed_size, self.embed_size)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        self.fc2 = nn.Linear(self.embed_size, self.embed_size)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, item_feats, outfit_map):
        all_outfit_feat = torch.index_select(item_feats, 0, outfit_map.contiguous().view(-1).long())
        all_outfit_feat = all_outfit_feat.view(-1, 2, self.embed_size)  # [n_outfits, 2, embed_dim]
        side_mess = torch.mul(all_outfit_feat[:,0,:], all_outfit_feat[:,1,:]) # [n_outfits, embed_dim]
        com_mess = self.relu(self.fc2(side_mess))
        ego_mess = self.relu(self.fc1(all_outfit_feat))
        return ego_mess, com_mess


class I2O(nn.Module):
    def __init__(self, embed_size, n_outfits, args):
        super(I2O, self).__init__()
        self.embed_size = embed_size
        self.n_outfits = n_outfits
        self.relu = nn.LeakyReLU()
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(self.embed_size, self.embed_size)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        self.bn = nn.BatchNorm1d(self.embed_size)

    def forward(self, ego_mess_i, com_mess_i):
        neigh = torch.sum(ego_mess_i, 1) + com_mess_i # [n_outfits, d]
        neigh_mess = self.relu(self.fc(neigh))
        neigh_mess = F.normalize(neigh_mess, p=2, dim=1)
        return neigh_mess


class O2U(nn.Module):
    def __init__(self, embed_size, n_users, n_outfits, layer_nums, args):
        super(O2U, self).__init__()
        self.embed_size = embed_size
        self.n_users = n_users
        self.n_outfits = n_outfits
        self.layer_nums = layer_nums
        self.relu = nn.LeakyReLU()
        self.drop = nn.Dropout(0.2)
        self.user_embedding = nn.Embedding(self.n_users, self.embed_size)
        self.user_embedding.weight.requires_grad = True
        self.outfit_embedding = nn.Embedding(self.n_outfits, self.embed_size)
        self.outfit_embedding.weight.requires_grad = True
        self.fc1 = nn.ModuleList([nn.Linear(self.embed_size, self.embed_size) for i in range(self.layer_nums)])
        for i in range(self.layer_nums):
            nn.init.xavier_uniform_(self.fc1[i].weight)
            nn.init.constant_(self.fc1[i].bias, 0)
        self.fc2 = nn.ModuleList([nn.Linear(self.embed_size, self.embed_size) for i in range(self.layer_nums)])
        for i in range(self.layer_nums):
            nn.init.xavier_uniform_(self.fc2[i].weight)
            nn.init.constant_(self.fc2[i].bias, 0)
        self.bn = nn.BatchNorm1d(self.embed_size)

    def forward(self, A_hat, o_embedding, u_id):
        u_embedding = self.user_embedding(u_id.long())
        ego_embeddings = torch.cat([u_embedding, o_embedding], 0)
        all_embeddings = ego_embeddings
        for i in range(self.layer_nums):
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            sum_embeddings = self.fc1[i](side_embeddings)
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = self.fc2[i](bi_embeddings)
            ego_embeddings = self.relu(sum_embeddings + bi_embeddings)
            ego_embeddings = self.drop(ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings = all_embeddings + norm_embeddings
        return all_embeddings


class MSA(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads):
        super(MSA, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.dim_split = self.dim_V // self.num_heads
        self.w_q = nn.Linear(dim_Q, dim_V)
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.constant_(self.w_q.bias, 0)
        self.w_k = nn.Linear(dim_K, dim_V)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.constant_(self.w_k.bias, 0)
        self.w_v = nn.Linear(dim_K, dim_V)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.constant_(self.w_v.bias, 0)
        self.fc = nn.Linear(dim_V, dim_V)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        self.ln0 = nn.LayerNorm(dim_V)
        self.ln1 = nn.LayerNorm(dim_V)
        self.scale = self.dim_split ** -0.5

    def forward(self, Q, K, V, mask):
        Q, K, V = self.w_q(Q).unsqueeze(0), self.w_k(K).unsqueeze(0), self.w_v(V).unsqueeze(0) # [1, 2 * n_c, d]
        Q_ = torch.cat(Q.split(self.dim_split, 2), 0) # [h, 2 * n_c, d/h]
        K_ = torch.cat(K.split(self.dim_split, 2), 0)
        V_ = torch.cat(V.split(self.dim_split, 2), 0)
        attention = Q_.bmm(K_.transpose(1, 2)) * self.scale # [h, 2 * n_c, 2 * n_c]
        A = torch.softmax(attention, dim=2)
        value = torch.matmul(A, V_) # [h, 2 * n_c, d/h]
        h = torch.cat(value.split(Q.size(0), 0), 2).squeeze(0) # [2 * n_c, d]
        h = self.ln0(h)
        f = self.ln1(h + F.relu(self.fc(h)))
        return f


class SetTransformer(nn.Module):
    def __init__(
            self,
            input_dim,
            embd_dim,
            num_heads,
            layer_RU_num,
            layer_RF_num
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embd_dim = embd_dim
        self.layer_RU_num = layer_RU_num
        self.layer_RF_num = layer_RF_num

        self.msa = nn.ModuleList([MSA(self.input_dim, self.input_dim, self.input_dim, num_heads) for i in range(self.layer_RU_num)])

        self.msa_tf = nn.ModuleList([MSA(self.input_dim, self.input_dim, self.input_dim, num_heads) for i in range(self.layer_RF_num)])
        self.msa_bf = nn.ModuleList([MSA(self.input_dim, self.input_dim, self.input_dim, num_heads) for i in range(self.layer_RF_num)])

        self.S = nn.Parameter(torch.Tensor(1, self.input_dim))
        nn.init.xavier_uniform_(self.S)


    def forward(self, x, mask):
        q0 = x # [2 * n_c, d]
        for i in range(self.layer_RU_num):
            q0 = self.msa[i](q0, q0, q0, mask)  # [2 * n_c, d]
        corpus_num = q0.shape[0]
        ql = q0.view(-1, 2, self.input_dim)
        ql = ql.permute(1, 0, 2)
        T = ql[0, :, :]
        B = ql[1, :, :]
        for i in range(self.layer_RF_num):
            if i == 0:
                T_F = torch.cat((self.S, T), 0)  # [1+n_c, d]
            else:
                T_F = torch.cat((F1, T), 0)
            T_F0 = self.msa_tf[i](T_F, T_F, T_F, mask)
            F0 = T_F0[0:1, :]
            B_F0 = torch.cat((F0, B), 0)
            B_F1 = self.msa_bf[i](B_F0, B_F0, B_F0, mask)
            F1 = B_F1[0:1, :]
        Z = F1
        return Z


class DP2Net(nn.Module):
    def __init__(self, args, n_users, n_outfits, n_items):
        super().__init__()

        self.args = args
        self.n_users = n_users
        self.n_outfits = n_outfits
        self.n_items = n_items
        self.input_size = self.args.input_size
        self.embed_size = self.args.embed_size
        self.drop_m = self.args.drop_m
        self.layer_AIP_num = self.args.layer_AIP_num
        self.layer_RU_num = self.args.layer_RU_num
        self.layer_RF_num = self.args.layer_RF_num
        self.num_heads = self.args.num_heads

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()

        self.mlp_embed = MLP(self.input_size, self.embed_size)
        self.i2i_encoder = I2I(self.embed_size, self.n_items, self.args)
        self.i2o_encoder = I2O(self.embed_size, self.n_outfits, self.args)
        self.o2u_encoder = O2U(self.embed_size, self.n_users, self.n_outfits, self.layer_AIP_num, self.args)
        self.corpus_visual_encoder = SetTransformer(self.embed_size, self.embed_size, self.num_heads, self.layer_RU_num, self.layer_RF_num)
        
    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()
        i = i[:, dropout_mask]
        v = v[dropout_mask]
        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, uid, pos_outfit_id, neg_outfit_id, visual_feat, L, outfit_map, corpus_feature, device, task):
        bsz = uid.shape[0]
        mask = None
        if task == 'train':
            drop_flag = True
        elif task == 'test':
            drop_flag = False
        L_hat = self.sparse_dropout(L, self.drop_m, L._nnz()) if drop_flag else L

        item_feats = self.mlp_embed(visual_feat)
        corpus_feature = self.mlp_embed(corpus_feature)

        ego_mess_i, com_mess_i = self.i2i_encoder(item_feats, outfit_map)

        o_embedding = self.i2o_encoder(ego_mess_i, com_mess_i)

        user_id_for_emb = torch.Tensor([i for i in range(self.n_users)]).to(device)
        e_embedding = self.o2u_encoder(L_hat, o_embedding, user_id_for_emb)

        u_feat = e_embedding[:self.n_users, :]
        o_feat = e_embedding[self.n_users:, :]
        
        user_feat = torch.index_select(u_feat, 0, uid.long())
        
        # user-general preference perception
        corpus_fusion_feat = self.corpus_visual_encoder(corpus_feature, mask)
        g_ug = torch.sum(torch.mul(corpus_fusion_feat, user_feat), 1, True)

        # user-specific preference perception
        if task == 'train':
            pos_outfit_feat = torch.index_select(o_feat, 0, pos_outfit_id)
            neg_outfit_feat = torch.index_select(o_feat, 0, neg_outfit_id)
            g_us_n = torch.sum(torch.mul(neg_outfit_feat, user_feat), 1, True)
            g_us_p = torch.sum(torch.mul(pos_outfit_feat, user_feat), 1, True)
            l_reg = ((torch.norm(user_feat) ** 2 + torch.norm(neg_outfit_feat) ** 2 + torch.norm(pos_outfit_feat) ** 2) / 2) / bsz
            return l_reg, g_ug, g_us_n, g_us_p
        
        elif task == 'test':
            outfit_feat = torch.index_select(o_feat, 0, neg_outfit_id)
            g_us = torch.sum(torch.mul(outfit_feat, user_feat), 1, True)
            return g_ug, g_us
        

            
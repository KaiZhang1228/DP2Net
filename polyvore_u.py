import os
import numpy as np
from torch.utils.data import DataLoader, Dataset


class PloyvoreDataset(Dataset):
    def __init__(self, args, split):
        super(PloyvoreDataset, self).__init__()
        self.split = split

        if self.split == 'train':
            self.train_file = os.path.join(args.data_path, args.dataset, 'u_pn_list.npy')
            self.train_list = np.load(self.train_file)
        if self.split == 'test':
            self.test_file = os.path.join(args.data_path, args.dataset, 'uo_list.npy')
            self.test_list = np.load(self.test_file)


    def __getitem__(self, index):
        if self.split == 'train':
            uid, pos_outfit_id, neg_outfit_id = self.train_list[index]
            return uid, pos_outfit_id, neg_outfit_id

        if self.split == 'test':
            uid, outfit_id, label = self.test_list[index]
            return uid, outfit_id, label


    def __len__(self):
        if self.split == 'train':
            return len(self.train_list)
        if self.split == 'test':
            return len(self.test_list)

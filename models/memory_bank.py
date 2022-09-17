import torch
import numpy as np


class MemoryBank(object):
    def __init__(self, embeddings_size, size):
        self.embeddings = torch.tensor(np.zeros((size, embeddings_size))).float()#.cuda()
        self.product_labels = torch.tensor(np.zeros(size)).float()#.cuda()
        self.size = size
        self.cur_size = 0

    def update(self, embeddings, product_labels):
        q_size = len(product_labels)
        if self.cur_size + q_size > self.size:
            self.embeddings[-q_size:] = embeddings
            self.product_labels[-q_size:] = product_labels
            self.cur_size = 0
        else:
            self.embeddings[self.cur_size: self.cur_size + q_size] = embeddings
            self.product_labels[self.cur_size: self.cur_size + q_size] = product_labels
            self.cur_size += q_size

    def get_embeddings(self):
        if self.product_labels[-1].item() != 0:
            return self.embeddings, self.product_labels
        else:
            return self.embeddings[:self.cur_size], self.product_labels[:self.cur_size]

import torch
import torch.nn as nn
import numpy as np
from configs.train_config import cfg as train_cfg


# class TripletLoss(nn.Module):
#     def __init__(self, margin=0.2):
#         super(TripletLoss, self).__init__()
#         self.margin = margin
#
#     def forward(self, inputs_col, targets_col, inputs_row, targets_row):
#         targets_col, targets_row = torch.tensor(targets_col), torch.tensor(targets_row)
#         b_size = inputs_col.size(0)
#         dists = torch.matmul(inputs_col, inputs_row.t())
#
#         p0 = targets_col.clone().view(1, targets_col.size()[0]).expand_as(dists)
#         p1 = targets_row.view(targets_row.size()[0], 1).expand_as(dists)
#
#         positives_ids = torch.eq(p0, p1).to(dtype=torch.uint8) - (torch.eye(len(dists)))#.to(self.device)
#         negatives_ids = (positives_ids == 0).to(dtype=torch.uint8) - (torch.eye(len(dists)))
#
#         losses = torch.tensor(0.0)
#         losses_ = []
#         for i in range(b_size):
#             pos_ids_ = np.atleast_1d(positives_ids[i].nonzero().squeeze().cpu().numpy())
#             neg_ids_ = np.atleast_1d(negatives_ids[i].nonzero().squeeze().cpu().numpy())
#
#             pos_dists = dists[i, pos_ids_]
#             neg_dists = dists[i, neg_ids_]
#
#             pos_pair_expanded = pos_dists.expand(len(neg_ids_), len(pos_ids_)).T
#             neg_pair_expanded = neg_dists.expand(len(pos_ids_), len(neg_ids_))
#             all_possible_ids = (pos_pair_expanded < neg_pair_expanded + self.margin).to(dtype=torch.uint8).nonzero().squeeze().cpu().numpy()
#             if len(all_possible_ids) > 0:
#                 pos_idxs, neg_idxs = (all_possible_ids[:, 0], all_possible_ids[:, 1]) if len(all_possible_ids.shape) > 1 \
#                     else (all_possible_ids[0], all_possible_ids[1])
#                 pos_dists = pos_dists[pos_idxs]
#                 neg_dists = neg_dists[neg_idxs]
#
#                 loss = torch.relu(pos_dists - neg_dists)
#                 losses += torch.sum(loss)
#                 if len(loss) > 0:
#                     losses_.extend(loss)
#
#         nb_non_zero_losses = torch.sum(torch.tensor(losses_) > 0)
#         loss = losses / nb_non_zero_losses
#         return loss


class TripletLoss(torch.nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    @staticmethod
    def get_distance(x1, x2):
        mm = torch.mm(x1, x2.t())
        dist = mm.diag().view((mm.diag().size()[0], 1))
        dist = dist.expand_as(mm)
        dist_ = dist + dist.t()
        dist_ = (dist_ - 2 * mm).clamp(min=0)
        return dist_.clamp(min=1e-4).sqrt()

    def sample_triplets(self, embeddings, prod_labels, embeddings1, prod_labels1):
        anchor_ids, pos_ids, neg_ids = [], [], []
        if not torch.is_tensor(prod_labels):
            prod_labels = torch.tensor(prod_labels)
            prod_labels1 = torch.tensor(prod_labels1)

        distance = self.get_distance(embeddings, embeddings1)
        p0 = prod_labels.clone().view(1, prod_labels.size()[0]).expand_as(distance)
        p1 = prod_labels1.view(prod_labels.size()[0], 1).expand_as(distance)
        # positives_ids = torch.eq(p0, p1).to(dtype=torch.float32).cuda() - (torch.eye(len(distance))).cuda()
        positives_ids = torch.eq(p0, p1).to(dtype=torch.float32) - (torch.eye(len(distance)))
        # n_ids = ((positives_ids > 0) + (distance < self.thr)).to(dtype=torch.float32).cuda()
        n_ids = (positives_ids > 0).to(dtype=torch.float32)

        negatives_ids = n_ids * 1e6 + distance
        to_retrieve_ids = max(1, min(int(positives_ids.data.sum()) // len(positives_ids), negatives_ids.size(1)))
        negatives = negatives_ids.topk(to_retrieve_ids, dim=1, largest=False)[1]
        negatives_ids_ = torch.zeros_like(negatives_ids.data).scatter(1, negatives, 1.0)

        for i in range(len(distance)):
            pos_ids_ = np.atleast_1d(positives_ids[i].nonzero().squeeze().cpu().numpy())
            neg_ids_ = np.atleast_1d(negatives_ids_[i].nonzero().squeeze().cpu().numpy())
            ids = distance[i][pos_ids_] < distance[i][neg_ids_] + self.margin
            pos_ids_ = pos_ids_[ids]
            neg_ids_ = neg_ids_[ids]
            if len(pos_ids_) > 0:
                anchor_ids.extend([i] * len(pos_ids_))
                pos_ids.extend(pos_ids_)
                neg_ids.extend(neg_ids_)

            # if len(anchor_ids) != len(pos_ids) or len(anchor_ids) != len(neg_ids):
            #     t = min(map(len, [anchor_ids, pos_ids, neg_ids]))
            #     anchor_ids = anchor_ids[:t]
            #     pos_ids = pos_ids[:t]
            #     neg_ids = neg_ids[:t]
        anchors, positives, negatives = embeddings[anchor_ids], embeddings[pos_ids], embeddings[neg_ids]
        return anchor_ids, anchors, positives, negatives

    def forward(self, embeddings, product_labels, embeddings1, prod_labels1):
        a_indices, anchors, positives, negatives= \
            self.sample_triplets(embeddings, product_labels, embeddings1, prod_labels1)

        d_ap = torch.sqrt(torch.sum((positives - anchors) ** 2, dim=1) + 1e-8)
        d_an = torch.sqrt(torch.sum((negatives - anchors) ** 2, dim=1) + 1e-8)

        pos_loss = torch.relu(d_ap + self.margin)
        neg_loss = torch.relu(- d_an + self.margin)

        loss = torch.sum(pos_loss + neg_loss)
        if int(torch.sum((pos_loss > 0.0) + (neg_loss > 0.0))) > 0:
            loss = loss / int(torch.sum((pos_loss > 0.0) + (neg_loss > 0.0)))
        return loss

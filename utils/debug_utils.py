import torch
import numpy as np

from configs.train_config import cfg as train_cfg
from utils.eval_utils import get_nearest_neighbors


def overfit_on_batch(dl, criterion, optimizer, model):
    batch = next(dl)
    images, labels = batch[0], batch[1]
    for iter_ in range(train_cfg.overfit_on_batch_iters):
        optimizer.zero_grad()
        model, images = model.cuda(), images.cuda()
        embeddings = model(images)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        loss = criterion(embeddings, labels, embeddings, labels)

        assert not torch.isnan(loss).any(), 'loss is nan'
        loss.backward()
        optimizer.step()
        top_1s = []
        for i, emb in enumerate(embeddings):
            neighbors = get_nearest_neighbors(torch.stack([em for em_i, em in enumerate(embeddings) if em_i != i]).detach().cpu().numpy().reshape(-1, 128),
                                              [l.item() for j, l in enumerate(labels) if j != i],
                                              emb.detach().cpu().numpy().reshape(-1, 128), k=1)
            top_1 = 1 if labels[i].item() == neighbors else 0
            top_1s.append(top_1)

        print(f'iter: {iter_}, loss: {loss.item()}, top 1: {np.mean(top_1s)}')
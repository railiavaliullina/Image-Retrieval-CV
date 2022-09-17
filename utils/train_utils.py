import torch

from configs.train_config import cfg as train_cfg
from losses.triplet_loss import TripletLoss


def get_optimizer(model):
    opt = torch.optim.Adam([
        {'params': model.parameters(), 'lr': train_cfg.lr, 'weight_decay': train_cfg.weight_decay}])
    return opt


def get_criterion():
    criterion = TripletLoss(train_cfg.margin).cuda()
    return criterion


def make_training_step(batch, criterion, optimizer, model, global_step, memory_bank):
    images, labels = batch[0], batch[1]
    embeddings = model(images)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    loss = criterion(embeddings, labels, embeddings, labels)

    if global_step > train_cfg.memory_bank_iter:
        print('sampling from bank')
        mb_enbeddings, mb_features = memory_bank.get_embeddings()
        mb_loss = criterion(embeddings, labels, mb_enbeddings, mb_features)
        loss = loss + mb_loss

    # loss = loss + mb_loss  # , embeddings, labels
    assert not torch.isnan(loss).any(), 'loss is nan'
    loss.backward()
    optimizer.step()
    return loss.item()

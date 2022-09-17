import torch
import time
import numpy as np
from tensorboardX import SummaryWriter
import tarfile

from data.dataset import get_dataloader
from models.resnet import get_model
from utils.train_utils import get_optimizer, get_criterion, make_training_step
from utils.eval_utils import visualize_embeddings
from utils.eval_utils import evaluate
from configs.train_config import cfg as train_cfg
from configs.eval_config import cfg as eval_cfg
from configs.dataset_config import cfg as dataset_cfg
from models.memory_bank import MemoryBank


def train():
    train_dataloader = get_dataloader(dataset_type='train')
    query_test_dataloader, retrieval_test_dataloader = get_dataloader(dataset_type='test')
    query_valid_dataloader, retrieval_valid_dataloader = get_dataloader(dataset_type='valid')

    model = get_model()
    optimizer = get_optimizer(model)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', min_lr=1e-8, patience=5, factor=0.8)
    criterion = get_criterion()

    if train_cfg.use_memory_bank:
        memory_bank = MemoryBank(embeddings_size=128, size=len(train_dataloader.dataset))
    else:
        memory_bank = None

    start_epoch, global_step = 0, -1

    # loading saved checkpoints if needed
    if train_cfg.continue_training_from_epoch:
        try:
            checkpoint = torch.load(train_cfg.checkpoints_dir + f'checkpoint_{train_cfg.checkpoint_from_epoch}.pth')
            model.load_state_dict(checkpoint['model'])
            start_epoch = checkpoint['epoch'] + 1
            global_step = checkpoint['global_step'] + 1
            optimizer.load_state_dict(checkpoint['opt'])
        except FileNotFoundError:
            print('Checkpoint not found')

    # evaluate before training if needed
    if eval_cfg.compute_metrics_before_training:
        model.eval()
        print(f'Evaluating on valid data..')
        top_1_valid_accuracy = evaluate(model, query_valid_dataloader, retrieval_valid_dataloader)
        print(f'Top-1 valid accuracy: {top_1_valid_accuracy}')
        print(f'Evaluating on test data..')
        top_1_test_accuracy = evaluate(model, query_test_dataloader, retrieval_test_dataloader)
        print(f'Top-1 test accuracy: {top_1_test_accuracy}')

        # visualize embeddings with tensorboard
        if eval_cfg.visualize_embeddings:
            embeddings_writer = SummaryWriter(log_dir=train_cfg.tensorboard_dir + f'/embeddings_vis/epoch_-1')
            visualize_embeddings(embeddings_writer, model, train_dataloader, num_batches=5, tag='training_batch_before_training')
        model.train()

    # main loop
    dl = iter(train_dataloader)
    for e in range(start_epoch, train_cfg.nb_epochs):
        print(f'Epoch: {e}/{train_cfg.nb_epochs}')
        epoch_start_time = time.time()
        embeddings_writer = SummaryWriter(log_dir=train_cfg.tensorboard_dir + f'/embeddings_vis/epoch_{e}')

        model.train()
        print('Starting training..')
        loss_list = []
        for i in range(len(train_dataloader)):
            batch = next(dl)
            loss = make_training_step(batch, criterion, optimizer, model, global_step, memory_bank)
            loss_list.append(loss)
            global_step += 1

            print(loss)

            if global_step % 50 == 0:
                if global_step != 0:
                    loss_mean = np.mean(loss_list[-50:])
                else:
                    loss_mean = loss
                print(f'global step: {global_step}, loss: {loss_mean}')

        # save checkpoints
        if train_cfg.save_model:
            print('Saving current model...')
            state = {
                'model': model.state_dict(),
                'epoch': e,
                'global_step': global_step,
                'opt': optimizer.state_dict()
                }
            torch.save(state, (train_cfg.checkpoints_dir + f'checkpoint_{e}.pth'))

        # evaluate model
        model.eval()
        print(f'Evaluating on valid data..')
        top_1_valid_accuracy = evaluate(model, query_valid_dataloader, retrieval_valid_dataloader)
        print(f'Top-1 valid accuracy: {top_1_valid_accuracy}')

        print(f'Evaluating on test data..')
        top_1_test_accuracy = evaluate(model, query_test_dataloader, retrieval_test_dataloader)
        print(f'Top-1 test accuracy: {top_1_test_accuracy}')
        model.train()

        # visualize embeddings with tensorboard
        if eval_cfg.visualize_embeddings:
            visualize_embeddings(embeddings_writer, model, train_dataloader, num_batches=5,
                                 tag='training_batch_during_training')

        print(f'epoch training time: {round((time.time() - epoch_start_time) / 60, 3)} min')


if __name__ == '__main__':
    if train_cfg.use_gpu:
        torch.cuda.set_device(0)
    np.random.seed(0)
    torch.manual_seed(0)

    total_training_start_time = time.time()
    train()
    print(f'Training time: {round((time.time() - total_training_start_time) / 60, 3)} min')

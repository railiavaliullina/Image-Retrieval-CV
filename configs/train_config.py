from easydict import EasyDict

cfg = EasyDict()

cfg.checkpoints_dir = 'D:/Users/Admin/PycharmProjects/imageretrievalvaliullina/'
cfg.tensorboard_dir = 'E:/'
cfg.device = 'cuda:0'
cfg.use_amp = False
cfg.nb_epochs = 100
cfg.continue_training_from_epoch = False
cfg.checkpoint_from_epoch = 0
cfg.batch_size = 64
cfg.lr = 1e-5
cfg.weight_decay = 1e-4
cfg.save_model = True
cfg.log_to_mlflow = True
cfg.use_gpu = False
cfg.margin = 0.2

cfg.train_on_kaggle = True
cfg.path_to_image_folders = 'E:/datasets/'

cfg.use_memory_bank = True
cfg.memory_bank_iter = 0  # 1000

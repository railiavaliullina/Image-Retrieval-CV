from easydict import EasyDict

cfg = EasyDict()
cfg.model = EasyDict()

cfg.model.architecture = 'resnet50'
cfg.model.layers = [3, 4, 6, 3]
cfg.model.progress = True
cfg.model.pretrained_model = True
cfg.model.embedding_dim = 128
cfg.model.random_seed = 0

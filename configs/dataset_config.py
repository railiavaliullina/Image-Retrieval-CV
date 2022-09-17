from easydict import EasyDict

cfg = EasyDict()

cfg.main_dataset_path = 'E:/datasets/'
cfg.sop_dataset_path = 'E:/datasets/Stanford_Online_Products'
cfg.dataset_split_pickle_file = 'E:/datasets/SOP_train_valid_test_split.pickle'
cfg.dataset_new_folder = 'E:/datasets/SOP_retrieval'
cfg.new_structure_info_pickle_file = 'E:/datasets/new_structure_info.pickle'
cfg.valid_query_retrieval_sets_path = 'E:/datasets/valid_dataset.pickle'
cfg.test_query_retrieval_sets_path = 'E:/datasets/test_dataset.pickle'

cfg.nb_categories = 12
cfg.sz_dataset = 120053
cfg.nb_elems_needed_for_product = 4

# augmentation
cfg.sz_crop = 224
cfg.sz_resize = 256
cfg.mean = [0.485, 0.456, 0.406]
cfg.std = [0.229, 0.224, 0.225]

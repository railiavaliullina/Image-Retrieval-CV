import pickle

from configs.dataset_config import cfg as dataset_cfg


def get_query_and_retrieval_sets(image_folder, dataset_type, type):

    query_retrieval_sets_path = dataset_cfg.valid_query_retrieval_sets_path if dataset_type == 'valid' \
        else dataset_cfg.test_query_retrieval_sets_path

    with open(query_retrieval_sets_path, 'rb') as f:
        query_retrieval_sets = pickle.load(f)

    set_paths = query_retrieval_sets[type]
    set_paths_and_labels = []

    for i, im in enumerate(image_folder.imgs):
        split = im[0].split("\\")
        img_path = '/'.join([im_ for im_ in split[1:]])
        if img_path in set_paths:
            set_paths_and_labels.append((split[0] + "/" + img_path, int(im[1])))

    return set_paths_and_labels

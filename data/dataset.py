from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from data.BatchSampler import BatchSampler
import numpy as np
import pickle
from collections import Counter
from copy import deepcopy

from utils.data_utils import get_query_and_retrieval_sets
from configs.dataset_config import cfg as dataset_cfg
from configs.train_config import cfg as train_config


def make_image_folder(dataset_type):
    """
    Создание Image Folder с соответствующей аугментацией
    """
    if dataset_type == 'train':
        transforms_ = transforms.Compose([
            transforms.RandomResizedCrop(dataset_cfg.sz_crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_cfg.mean, std=dataset_cfg.std)
        ])
    else:
        transforms_ = transforms.Compose([
            transforms.Resize(dataset_cfg.sz_resize),
            transforms.CenterCrop(dataset_cfg.sz_crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_cfg.mean, std=dataset_cfg.std)
        ])
    image_folder = datasets.ImageFolder(root=f'{dataset_cfg.dataset_new_folder}/{dataset_type}', transform=transforms_)

    if dataset_type == 'train':
        # get category labels and dict {category_1: list of category_1 products, ...}
        categories_to_products = {k: [] for k in range(dataset_cfg.nb_categories)}
        product_labels = [im[1] for im in image_folder.imgs]
        product_labels_counter = Counter(product_labels)
        product_labels_to_keep = [k for k, v in product_labels_counter.items() if v >= dataset_cfg.nb_elems_needed_for_product]
        ids_to_keep = np.asarray([i for i, p in enumerate(product_labels) if p in product_labels_to_keep])
        product_labels = list(np.asarray(product_labels)[ids_to_keep])
        image_folder.imgs = list(np.asarray(image_folder.imgs)[ids_to_keep])

        for i, im in enumerate(image_folder.imgs):
            category_label = int(im[0].split('\\')[1].split('_')[0])
            assert int(im[1]) == product_labels[i]
            categories_to_products[category_label].append(product_labels[i])
        categories_to_products = {k: np.unique(v) for k, v in categories_to_products.items()}
        image_folder.category_labels = [int(im[0].split('\\')[1].split('_')[0]) for im in image_folder.imgs]
        image_folder.categories_to_products = categories_to_products

    # для обучения на Kaggle
    with open(f'E:/datasets/image_folder_{dataset_type}', 'wb') as f:
        pickle.dump(image_folder, f)
    return image_folder


def get_dataloader(dataset_type):
    print(f'Getting {dataset_type} dataloader..')
    if train_config.train_on_kaggle:
        with open(train_config.path_to_image_folders + f'image_folder_{dataset_type}', 'rb') as f:
            image_folder = pickle.load(f)
    else:
        image_folder = make_image_folder(dataset_type)
    if dataset_type == 'train':
        batch_sampler = BatchSampler(image_folder, train_config.batch_size) if dataset_type == 'train' else None
        dataloader = DataLoader(image_folder, batch_sampler=batch_sampler)
        return dataloader

    query_image_folder = deepcopy(image_folder)
    retrieval_image_folder = deepcopy(image_folder)
    imgs = get_query_and_retrieval_sets(image_folder, dataset_type, type='query')
    query_image_folder.imgs = imgs
    query_image_folder.samples = imgs
    query_dataloader = DataLoader(query_image_folder, batch_size=train_config.batch_size)

    imgs = get_query_and_retrieval_sets(image_folder, dataset_type, type='retrieval')
    retrieval_image_folder.imgs = imgs
    retrieval_image_folder.samples = imgs
    retrieval_dataloader = DataLoader(retrieval_image_folder, batch_size=train_config.batch_size)
    return (query_dataloader, retrieval_dataloader)

import pickle
import numpy as np
from shutil import copyfile
from torchvision import transforms, datasets
import os
import time
import pandas as pd

from configs.dataset_config import cfg as dataset_cfg


class PrepareDataset(object):
    """
    Класс для считывания данных c формированием частей на основе pickle файла
    и сохранения данных с новой структурой
    """

    def __init__(self, cfg, dataset_type):
        """
        :param cfg: конфиг с параметрами для датасета
        :param dataset_type: тип данных ('train', 'test', 'valid'),
        которые необходимо считать и сохранить с новой структурой
        """
        self.cfg = cfg
        self.dataset_path = cfg.sop_dataset_path
        self.dataset_type = dataset_type
        self.category_labels, self.paths, self.product_labels, self.new_paths = [], [], [], []

        # разбиение на части на основе pickle файла (в зависимости от dataset_type)
        with open(cfg.dataset_split_pickle_file, 'rb') as f:
            dataset_split_pickle_file = pickle.load(f)

        for category, category_values in dataset_split_pickle_file.items():
            data_type_values = category_values[self.dataset_type]
            self.paths.extend(data_type_values['paths'])
            self.product_labels.extend(data_type_values['product_labels'])
            self.category_labels.extend(data_type_values['category_labels'])
        self.indexes = np.arange(len(self.product_labels))

        # сохранение данных с новой структурой для текущей части данных
        self.save_data_with_new_structure()

        # создание image folder
        self.make_image_folder()

    def __len__(self):
        """
        :return: размер текущей части ('train', 'test', 'valid') данных
        """
        return len(self.product_labels)

    @staticmethod
    def make_new_dirs(paths):
        """
        Создание новых директорий, если таких еще нет пот сохранении изображений по новой структуре
        :param paths: пути, по которым необходимо создать директории
        """
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

    def save_new_structure_info(self):
        """
        Сохранение новых путей и меток классов в pickle файл
        :return:
        """
        df = pd.DataFrame()
        df['paths'] = self.new_paths
        df['category_labels'] = self.category_labels
        df['product_labels'] = self.product_labels
        df.to_pickle(self.cfg.new_structure_info_pickle_file)

    def save_data_with_new_structure(self):
        """
        Копирование изображений из предыдущих местоположений в новые
        """
        self.make_new_dirs([self.cfg.dataset_new_folder,
                            os.path.join(self.cfg.dataset_new_folder, self.dataset_type)])

        for i, path in enumerate(self.paths):
            old_path = self.cfg.main_dataset_path + self.paths[i]
            product_label = self.product_labels[i]
            class_name, filename = self.paths[i].split('/')[2], self.paths[i].split('/')[3]
            new_path = f'{self.cfg.dataset_new_folder}/{self.dataset_type}/{product_label}/{class_name}/{class_name}_{filename}'
            self.make_new_dirs([f'{self.cfg.dataset_new_folder}/{self.dataset_type}/{product_label}',
                                f'{self.cfg.dataset_new_folder}/{self.dataset_type}/{product_label}/{class_name}'])
            try:
                copyfile(src=old_path, dst=new_path)
            except:
                print(old_path)
                # pass
            self.new_paths.append(
                f'SOP_retrieval/{self.dataset_type}/{product_label}/{class_name}/{class_name}_{filename}')

        self.save_new_structure_info()

    def make_image_folder(self):
        """
        Создание Image Folder с соответствующей аугментацией
        """
        if self.dataset_type == 'train':
            transforms_ = transforms.Compose([
                transforms.RandomResizedCrop(dataset_cfg.sz_crop),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=dataset_cfg.mean, std=dataset_cfg.std)
            ])
        else:
            transforms_ = transforms.Compose([
                transforms.Resize(self.cfg.sz_resize),
                transforms.CenterCrop(self.cfg.sz_crop),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.cfg.mean, std=self.cfg.std)
            ])
        self.image_folder = datasets.ImageFolder(
            root=f'{dataset_cfg.dataset_new_folder}/{self.dataset_type}',
            transform=transforms_)


if __name__ == '__main__':
    start_time = time.time()
    train_set = PrepareDataset(dataset_cfg, dataset_type='train')
    test_set = PrepareDataset(dataset_cfg, dataset_type='test')
    valid_set = PrepareDataset(dataset_cfg, dataset_type='valid')

    # проверка, соответвствуют ли размеры считанных данных суммарному количеству изображений в SOP
    dataset_size = len(train_set) + len(test_set) + len(valid_set)
    assert dataset_size == dataset_cfg.sz_dataset, 'incorrect dataset size'
    print(f'Dataset structuring time: {round((time.time() - start_time) / 60, 3)} min')

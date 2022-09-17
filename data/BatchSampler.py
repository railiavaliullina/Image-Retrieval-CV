import torch
import numpy as np


class BatchSampler(torch.utils.data.sampler.BatchSampler):
    def __init__(self, dataset, batch_size, n=4, m=4, l=4):
        self.n = n  # категорий
        self.m = m  # продуктов
        self.l = l  # изображений
        self.dataset = dataset
        self.batch_size = batch_size

        self.category_labels = np.array(dataset.category_labels)
        self.unique_category_labels = list(set(self.category_labels))

        self.prod_labels = np.array(dataset.targets)
        self.unique_prod_labels = list(set(self.prod_labels))

        self.current_product_label_indices = self.get_all_labels_indices_with_current_label(
            unique_labels=self.unique_prod_labels,
            labels=self.prod_labels)

    @staticmethod
    def get_all_labels_indices_with_current_label(unique_labels, labels):
        out = []
        # для каждого уникального продукта/категории запоминаем все индексы
        for c in unique_labels:
            prod_label_indices = np.where(labels == c)[0]
            np.random.shuffle(prod_label_indices)
            out.append(prod_label_indices)
        return out

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        for _ in range(len(self.dataset) // self.batch_size):
            assert len(self.unique_category_labels) >= self.n, 'not enough categories'
            chosen_categories = np.random.choice(len(self.unique_category_labels), self.n, replace=False)
            chosen_categories_products = [self.dataset.categories_to_products[category] for category in chosen_categories]

            chosen_products = []
            for category_products in chosen_categories_products:
                chosen_products.extend(np.random.choice(category_products, self.m, replace=False))

            chosen_ids = []
            for product in chosen_products:
                cur_product_ids = self.current_product_label_indices[product]
                assert len(cur_product_ids) >= self.l, 'not enough images'
                chosen_ids.extend(np.random.choice(cur_product_ids, self.l, replace=False))

            assert len(chosen_ids) > 0
            yield chosen_ids

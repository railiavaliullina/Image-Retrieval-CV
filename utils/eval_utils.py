import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def get_top_k(product_labels, neighbours_ids, cur_k):
    sum_ = 0
    for query_lbl, nearest_ids in zip(product_labels, neighbours_ids):
        if query_lbl in nearest_ids[:cur_k]:
            sum_ += 1
    recall_at_k = sum_ / len(product_labels)
    return recall_at_k


def get_nearest_neighbors(retrieval_embeddings, query_embeddings, k=1):
    index = NearestNeighbors(n_neighbors=k)
    index.fit(retrieval_embeddings)
    neighbors = index.kneighbors(query_embeddings)
    print(neighbors.shape)
    print(neighbors)
    return neighbors


def evaluate(model, query_dl, retrieval_dl):
    query_embeddings, query_labels = compute_embeddings(model, query_dl)
    retrieval_embeddings, retrieval_labels = compute_embeddings(model, retrieval_dl)

    neighbors = get_nearest_neighbors(retrieval_embeddings, query_embeddings)
    n_neighbors = np.array([[retrieval_labels[i] for i in ii] for ii in neighbors])
    top_1 = get_top_k(query_labels, n_neighbors, cur_k=1) * 100
    return top_1


def compute_embeddings(model, dl):
    print('Computing embeddings..')
    dl_len = len(dl)
    dl = iter(dl)
    all_embeddings, all_labels = [], []
    for i in range(dl_len):
        # if i % 50 == 0:
        print(f'iter: {i}/{dl_len}')
        x, y = next(dl)
        embeddings = model(x)
        all_labels.extend(y.data.cpu().numpy())
        all_embeddings.extend(embeddings.data.cpu().numpy())
    return all_embeddings, all_labels


def visualize_embeddings(embeddings_writer, model, dataloader, num_batches=5, tag=''):
    all_embeddings, all_labels = [], []
    dl = iter(dataloader)
    for i in range(num_batches):
        batch = next(dl)
        images, labels = batch[0], batch[1]
        embeddings = model(images)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        all_embeddings.extend(embeddings)
        all_labels.extend(labels)

    embeddings_writer.add_embedding(np.asarray([e.detach().numpy() for e in all_embeddings]),
                                    metadata=np.asarray([a.item() for a in all_labels]), tag=tag)

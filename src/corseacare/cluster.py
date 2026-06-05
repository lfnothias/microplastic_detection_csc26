import numpy as np


def standardize(X) -> np.ndarray:
    X = np.asarray(X, float)
    return (X - X.mean(0)) / (X.std(0) + 1e-8)


def hybrid_matrix(feat_vecs, embeddings, feat_weight=1.0, emb_weight=1.0) -> np.ndarray:
    F = feat_weight * standardize(feat_vecs)
    if embeddings is None or len(embeddings) == 0:
        return F
    E = emb_weight * standardize(embeddings)
    return np.hstack([F, E])


def cluster_hdbscan(X, min_cluster_size=5) -> np.ndarray:
    from sklearn.cluster import HDBSCAN
    return HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean").fit_predict(np.asarray(X))


def group_by_image(meta) -> dict:
    groups = {}
    for i, m in enumerate(meta):
        groups.setdefault(m["image"], []).append(i)
    return groups

import numpy as np
from corseacare.cluster import standardize, hybrid_matrix, cluster_hdbscan, group_by_image


def test_standardize_zero_mean_unit_var():
    X = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]])
    Z = standardize(X)
    assert np.allclose(Z.mean(0), 0, atol=1e-6)
    assert np.allclose(Z.std(0), 1, atol=1e-6)


def test_hybrid_matrix_concats_dims():
    feats = np.zeros((5, 7)); emb = np.zeros((5, 384))
    assert hybrid_matrix(feats, emb).shape == (5, 7 + 384)
    assert hybrid_matrix(feats, None).shape == (5, 7)


def test_hdbscan_finds_two_blobs():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.05, (30, 4)); b = rng.normal(5, 0.05, (30, 4))
    labels = cluster_hdbscan(np.vstack([a, b]), min_cluster_size=5)
    assert len({l for l in labels if l >= 0}) == 2


def test_group_by_image():
    meta = [{"image": "a"}, {"image": "b"}, {"image": "a"}]
    assert group_by_image(meta) == {"a": [0, 2], "b": [1]}

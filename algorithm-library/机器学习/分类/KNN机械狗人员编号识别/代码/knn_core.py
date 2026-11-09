"""Single digit classification using exact distance-weighted neighbors."""
import numpy as np


def neighbors(train, query, k=3, batch=128):
    train = np.asarray(train, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    if train.ndim != 2 or query.ndim != 2:
        raise ValueError('expected two-dimensional feature matrices')
    if train.shape[1] != query.shape[1] or train.shape[1] == 0:
        raise ValueError('feature dimensions must match and be positive')
    if type(k) is not int or not 1 <= k <= len(train):
        raise ValueError('k must be an integer in [1, training size]')
    if type(batch) is not int or batch < 1:
        raise ValueError('batch must be a positive integer')
    if not np.isfinite(train).all() or not np.isfinite(query).all():
        raise ValueError('features must be finite')
    indices = np.empty((len(query), k), dtype=np.int64)
    distances = np.empty((len(query), k), dtype=np.float64)
    norm = (train * train).sum(axis=1)
    for lo in range(0, len(query), batch):
        q = query[lo:lo + batch]
        squared = (q*q).sum(axis=1)[:, None] + norm[None, :]
        squared -= 2 * (q @ train.T)
        np.maximum(squared, 0, out=squared)
        # Stable sorting resolves equal distances by training row order.
        order = np.argsort(squared, axis=1, kind='stable')[:, :k]
        indices[lo:lo + len(q)] = order
        distances[lo:lo + len(q)] = np.sqrt(
            np.take_along_axis(squared, order, axis=1))
    return indices, distances


def vote(labels, indices, distances, weighted=True):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('labels must be a one-dimensional integer array')
    if len(labels) == 0 or labels.min() < 0 or labels.max() > 9:
        raise ValueError('labels must lie in 0..9')
    if indices.shape != distances.shape or indices.ndim != 2:
        raise ValueError('neighbor arrays must have the same 2D shape')
    if indices.shape[1] == 0 or (indices < 0).any():
        raise ValueError('empty or negative neighbor indices')
    if (indices >= len(labels)).any() or not np.isfinite(distances).all():
        raise ValueError('invalid neighbor index or distance')
    if (distances < 0).any():
        raise ValueError('distances must be nonnegative')
    scores = np.zeros((len(indices), 10), dtype=np.float64)
    for i, (ix, ds) in enumerate(zip(indices, distances)):
        if weighted:
            zero = ds == 0
            weights = zero.astype(float) if zero.any() else 1 / ds
        else:
            weights = np.ones(len(ds))
        scores[i] = np.bincount(labels[ix], weights=weights, minlength=10)
    prediction = scores.argmax(axis=1)  # Equal scores: smallest digit.
    support = scores.max(axis=1) / scores.sum(axis=1)
    return prediction, support


def recognize(train, labels, query, k=3, batch=128, weighted=True):
    if len(train) != len(labels):
        raise ValueError('training samples and labels must have equal length')
    indices, distances = neighbors(train, query, k, batch)
    prediction, support = vote(labels, indices, distances, weighted)
    return prediction, support, indices, distances

import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
from zipfile import ZipFile
from time import perf_counter
import csv
import hashlib
import json
import platform
import numpy as np
from knn_core import neighbors, vote, recognize

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = ROOT / 'results'
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)


def load(name):
    values = np.loadtxt(DATA / name, delimiter=',', dtype=int)
    assert values.shape[1] == 65
    assert ((values[:, :64] >= 0) & (values[:, :64] <= 16)).all()
    assert ((values[:, 64] >= 0) & (values[:, 64] <= 9)).all()
    return values[:, :64].astype(float) / 16, values[:, 64]


def split(labels, seed=42):
    rng = np.random.default_rng(seed)
    fit, valid = [], []
    for c in range(10):
        ix = np.flatnonzero(labels == c)
        rng.shuffle(ix)
        count = int(np.floor(len(ix) * 0.2))
        valid.extend(ix[:count])
        fit.extend(ix[count:])
    return np.sort(fit), np.sort(valid)


def metrics(y, pred):
    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(10),
                          where=cm.sum(0) != 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(10),
                       where=cm.sum(1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(10), where=precision + recall != 0)
    return dict(accuracy=float((y == pred).mean()),
                macro_f1=float(f1.mean()), errors=int((y != pred).sum()),
                recall=recall.tolist(), confusion=cm.tolist())


def self_check():
    rng = np.random.default_rng(8)
    a, q = rng.normal(size=(20, 4)), rng.normal(size=(6, 4))
    ix, ds = neighbors(a, q, 5, 2)
    direct = np.linalg.norm(q[:, None, :] - a[None, :, :], axis=2)
    expected = np.argsort(direct, axis=1, kind='stable')[:, :5]
    assert np.array_equal(ix, expected)
    assert np.allclose(ds, np.take_along_axis(direct, ix, 1))
    # Zero distances vote only among the matching samples.
    p, s = vote(np.array([7, 7, 2]), np.array([[0, 1, 2]]),
                np.array([[0., 0., 1.]]))
    assert p[0] == 7 and s[0] == 1
    p, _ = vote(np.array([7, 2]), np.array([[0, 1]]),
                np.array([[1., 1.]]))
    assert p[0] == 2
    for args in [(a, q, 0, 2), (a, q[:, :3], 1, 2),
                 (a, q, 1, 0), (a, q * np.nan, 1, 2)]:
        try:
            neighbors(*args)
        except ValueError:
            continue
        raise AssertionError('bad input accepted')
    return 'direct-distance equivalence, zero-distance, tie, 4 invalid inputs passed'


def main():
    archive = ROOT / 'optdigits.zip'
    if not (DATA / 'optdigits.tra').exists():
        with ZipFile(archive) as z:
            for name in ('optdigits.tra', 'optdigits.tes', 'optdigits.names'):
                (DATA / name).write_bytes(z.read(name))
    x, y = load('optdigits.tra')
    xt, yt = load('optdigits.tes')
    fi, va = split(y)
    np.savez(OUT / 'split_indices.npz', fit=fi, validation=va)
    ix, ds = neighbors(x[fi], x[va], 9)
    validation = []
    for k in (1, 3, 5, 7, 9):
        pred, _ = vote(y[fi], ix[:, :k], ds[:, :k])
        validation.append(dict(k=k, **metrics(y[va], pred)))
    # Highest validation accuracy, then smaller k; test set unused here.
    best = max(validation, key=lambda r: (r['accuracy'], -r['k']))['k']
    centroids = np.array([x[y == c].mean(0) for c in range(10)])
    methods = {
        'centroid': lambda: neighbors(centroids, xt, 1)[0][:, 0],
        'uniform': lambda: recognize(x, y, xt, best, weighted=False)[0],
        'weighted': lambda: recognize(x, y, xt, best)[0]}
    test_results, predictions = {}, {}
    for name, fn in methods.items():
        fn()  # Warm-up excluded from timing; parameters already frozen.
        times = []
        for _ in range(5):
            start = perf_counter()
            pred = fn()
            times.append((perf_counter() - start) * 1000)
        predictions[name] = pred
        test_results[name] = dict(**metrics(yt, pred),
            time_mean_ms=float(np.mean(times)), time_sd_ms=float(np.std(times)))
    pred, support, ix, ds = recognize(x, y, xt, best)
    errors = np.flatnonzero(pred != yt)
    with (OUT / 'predictions.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['test_row_1based', 'true', 'predicted', 'vote_support'])
        w.writerows((i+1, int(yt[i]), int(pred[i]), float(support[i]))
                    for i in range(len(yt)))
    change = predictions['uniform'] != pred
    demo = dict(test_row_1based=1, true=int(yt[0]), predicted=int(pred[0]),
                support=float(support[0]),
                neighbor_rows_1based=(ix[0]+1).tolist(),
                neighbor_labels=y[ix[0]].tolist(), distances=ds[0].tolist())
    # The drawing script uses this artifact; prediction logic has no PIL dependency.
    np.savez(OUT / 'visual_data.npz', images=xt.reshape(-1, 8, 8),
             true=yt, pred=pred, errors=errors, support=support)
    report = dict(python=platform.python_version(), numpy=np.__version__,
        platform=platform.platform(), cpu=platform.processor(),
        training=len(y), fitting=len(fi), validation_count=len(va), test=len(yt),
        chosen_k=best, validation=validation, results=test_results,
        tests=self_check(), demo=demo, error_rows_1based=(errors+1).tolist(),
        changed_by_weighting=int(change.sum()),
        fixed_by_weighting=int((change & (pred == yt)).sum()),
        broken_by_weighting=int((change & (predictions['uniform'] == yt)).sum()),
        hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (DATA / 'optdigits.tra', DATA / 'optdigits.tes')})
    (OUT / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

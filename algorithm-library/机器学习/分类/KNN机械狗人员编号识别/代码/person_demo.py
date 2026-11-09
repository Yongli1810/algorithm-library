"""Demonstrate digit-badge lookup; identity labels are fictional."""
from pathlib import Path
import json
import numpy as np
from knn_core import recognize

ROOT = Path(__file__).resolve().parent
REGISTRY = {digit: f'P{digit:02d}' for digit in range(10)}


def identify_badge(train, labels, rectified_digit, k=3):
    pixels = np.asarray(rectified_digit, dtype=float)
    if pixels.shape != (8, 8) or not np.isfinite(pixels).all():
        raise ValueError('require an already normalized 8x8 digit badge')
    if pixels.min() < 0 or pixels.max() > 16:
        raise ValueError('pixel counts must be in 0..16')
    pred, support, indices, distances = recognize(
        train, labels, pixels.reshape(1, 64)/16, k=k)
    digit = int(pred[0])
    return dict(digit=digit, registered_id=REGISTRY[digit],
                vote_support=float(support[0]),
                neighbor_rows=(indices[0]+1).tolist(),
                distances=distances[0].tolist())


if __name__ == '__main__':
    training = np.loadtxt(ROOT/'data/optdigits.tra', delimiter=',')
    testing = np.loadtxt(ROOT/'data/optdigits.tes', delimiter=',')
    result = identify_badge(training[:, :64]/16,
                            training[:, 64].astype(int),
                            testing[0, :64].reshape(8, 8))
    assert result['registered_id'] == 'P00'
    print(json.dumps(result, indent=2))

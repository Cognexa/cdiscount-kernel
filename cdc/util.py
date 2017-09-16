import io
import hashlib
import logging
from itertools import islice
from collections import defaultdict

import skimage.data
import numpy as np
try:
    import cv2
except ImportError:
    logging.info('Can not import OpenCV')


def sha256(filename: str):
    """Compute sha256 hash of the given file."""
    hash_sha256 = hashlib.sha256()
    with open(filename, 'rb') as file_:
        for chunk in iter(lambda: file_.read(4096), b''):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def gen_batch(iterable, batch_size):
    """Transform example iteration to batch iteration with the given batch_size."""
    source = iter(iterable)
    while True:
        chunk = list(islice(source, batch_size))
        if not chunk:
            raise StopIteration
        yield chunk


def decode_skimage(bytes):
    """Decode jpeg image with skimage back-end."""
    return skimage.data.imread(io.BytesIO(bytes))


def decode_cvimage(bytes):
    """Decode jpeg image with opencv back-end."""
    return cv2.imdecode(np.frombuffer(bytes, dtype=np.uint8), cv2.IMREAD_COLOR)


decode_image = decode_skimage
# alternatively, use opencv
# decode_image = decode_cvimage


def major_voting(ids, predictions):
    """Compute major vote for products with multiple images. If tied, use the 1st one."""
    product_to_prediction = defaultdict(list)
    for id, prediction in zip(ids, predictions):
        product_to_prediction[id].append(prediction)
    for id in product_to_prediction:
        voting = product_to_prediction[id]
        product_to_prediction[id] = sorted(zip(voting, [voting.count(vote) for vote in voting]),
                                           key=lambda x: -x[1])[0][0]
    return zip(*product_to_prediction.items())

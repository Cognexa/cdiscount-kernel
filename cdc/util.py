import hashlib
from itertools import islice


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

import os
import logging
import random
import os.path as path
from itertools import takewhile

import bson
import cxflow as cx
import pandas as pd

from .util import gen_batch, sha256, decode_image


class CDCNaiveDataset(cx.datasets.BaseDataset):
    """
    Simple dataset for <https://www.kaggle.com/c/cdiscount-image-classification-challenge>.

    Featuring:
        - cli download
        - validation
        - simple visualization
        - train-valid split
        - low memory footprint data streams

    See ../README.md for more info.
    """

    FILENAMES = ['category_names.7z', 'sample_submission.7z', 'train_example.bson', 'test.bson', 'train.bson']
    """Filenames as available for download."""

    EXTRACTED = ['category_names.csv', 'sample_submission.csv', 'train_example.bson', 'test.bson', 'train.bson']
    """Filenames after extracting 7z archives."""

    HASHES = ['84fe1e7334836d50ed04d475cfc525bccbe420f7242f85ca301b3f69294632c6',
              'a4ea875b408504bb9e981a7462a41f7d03cc0f68eecc8b222ecf0afc8e43e688',
              '5d54291c3704a755178d9c1cd8f877eaa6adbf207713988ca2bb5cd52aab7bdb',
              '844d3e13fa785498c2b153bc0edc942d14bbc95b92f30c827487ef096fd28a53',
              '2b9ac4157e67fc96ab85ca99679b3b25cada589c4da6bb128fa006085b4cc42b']
    """SHA256 hashes of EXTRACTED files (see <https://www.kaggle.com/blazeka/validate-download-with-sha256-hash>)."""

    TEST_FILE = 'test.bson'
    """Test data file."""

    TRAIN_FILE = 'train.bson'
    """Train data file. May be changed to `train_examples.bson` for a quick check."""

    CATEGORIES_FILE = 'categories.csv'
    """CSV file with (category_id -> integer class) mapping."""

    VISUAL_DIR = 'visual'
    """Dir to save some images with `cxflow dataset show cdc`."""

    def _configure_dataset(self, data_root: str='data',
                           split_file: str='split.csv',
                           valid_percent: float=0.1,
                           batch_size: int=64,
                           **kwargs):
        self._data_root = data_root
        self._split_file = split_file
        self._valid_percent = valid_percent
        self._batch_size = batch_size
        self._split = None
        self._categories = None

    def download(self):
        """
        Download and extract the data with kaggle-cli <https://github.com/floydwch/kaggle-cli>

        Use KG_USER and KG_PASS env. variables for log-in.

        Run with `KG_USER="<YOUR KAGGLE USERNAME" KG_PASS="<YOUR KAGGLE PASSWORD>" cxflow dataset download cdc`
        """
        os.makedirs(self._data_root, exist_ok=True)
        os.chdir(self._data_root)

        # download the data
        user = os.getenv('KG_USER')
        password = os.getenv('KG_PASS')
        if not user or not password:
            logging.error('Set up KG_USER and KG_PASS env variables in order to download the data')
        else:
            logging.info('Downloading data with kaggle-cli to `%s`', self._data_root)
            os.system('kg download -u "{}" -p "{}" -c "cdiscount-image-classification-challenge"'.format(user, password))

        # extract archives
        for filename in self.FILENAMES:
            if filename.endswith('7z') and path.exists(filename):
                logging.info('Extracting `%s`', filename)
                os.system('7z e -aoa {}'.format(filename))

    def validate(self):
        """
        Validate the downloaded data.

        Run with `cxflow dataset validate cdc`
        """
        logging.info('Validating file hashes, this may take a few minutes')
        for filename, hash_ in zip(self.EXTRACTED, self.HASHES):
            filepath = path.join(self._data_root, filename)
            if not path.exists(filepath):
                logging.warning('%s: missing', filename)
                continue
            computed_hash = sha256(filepath)
            if computed_hash == hash_:
                logging.info('%s: OK', filename)
            else:
                logging.error('%s: fail', filename)
                logging.error('expected: %s', hash_)
                logging.error('computed: %s', computed_hash)

    def show(self):
        """
        Save a few images from the TRAIN_FILE.

        For more elaborate visualizations and statistics, see other kernels such as
        <https://www.kaggle.com/bguberfain/just-showing-a-few-images>
        <https://www.kaggle.com/bguberfain/naive-statistics>

        Run with `cxflow dataset show cdc`
        """
        os.makedirs(self.VISUAL_DIR, exist_ok=True)
        data = bson.decode_file_iter(open(path.join(self._data_root, self.TRAIN_FILE), 'rb'))
        for i, example in takewhile(lambda x: x[0] < 10, enumerate(data)):
            for j, image in enumerate(example['imgs']):
                with open(path.join(self.VISUAL_DIR, 'image_{}_{}.jpg'.format(i, j)), 'wb') as file_:
                    file_.write(image['picture'])
        logging.info('Images saved to `%s`', self.VISUAL_DIR)

    def split(self):
        """
        Split train data to train and validation sets and compute (category_id -> integer class) mapping.

        Run with `cxflow dataset split cdc`
        :return:
        """
        # read example headers
        logging.info('Reading examples metadata, this may take a minute or two')
        ids = []
        categories = []
        for example in bson.decode_file_iter(open(path.join(self._data_root, self.TRAIN_FILE), 'rb')):
            ids.append(example['_id'])
            categories.append(example['category_id'])

        # generate random split
        size = len(ids)
        train_size = int(size*(1-self._valid_percent))
        valid_size = (size-train_size)
        split = ['train']*train_size + ['valid']*valid_size
        random.shuffle(split)

        # save split
        split_df = pd.DataFrame({'id': ids, 'split': split})
        split_path = path.join(self._data_root, self._split_file)
        split_df.to_csv(split_path, index=False)
        logging.info('Split train-valid of size %s-%s was written to `%s`', train_size, valid_size, split_path)

        # save (category_id -> integer class) mapping
        categories = sorted(list(set(categories)))
        categories_df = pd.DataFrame({'category_id': categories, 'class': list(range(len(categories)))})
        categories_path = path.join(self._data_root, self.CATEGORIES_FILE)
        categories_df.to_csv(categories_path, index=False)
        logging.info('Categories mapping saved to `{}`'.format(categories_path))

    def _load_meta(self):
        if self._split is None:
            logging.info('Loading metadata')
            self._split = pd.read_csv(path.join(self._data_root, self._split_file), index_col=0)
            self._categories = pd.read_csv(path.join(self._data_root, self.CATEGORIES_FILE), index_col=0)

    def _produce_examples(self, name: str):
        """
        Iterate through training images and produce (image, class) pairs

        You may augment the data here.
        Tip: use opencv backend (see .util.py) and resize the image to say 64x64.
        """
        self._load_meta()
        for example in bson.decode_file_iter(open(path.join(self._data_root, self.TRAIN_FILE), 'rb')):
            if self._split.loc[example['_id']]['split'] == name:
                for image in example['imgs']:
                    yield (decode_image(image['picture'])/255, self._categories.loc[example['category_id']]['class'])

    def _produce_predict_examples(self):
        """Iterate through test images and produce (image, _id) pairs"""
        for example in bson.decode_file_iter(open(path.join(self._data_root, self.TEST_FILE), 'rb')):
            for image in example['imgs']:
                yield (decode_image(image['picture'])/255, example['_id'])

    def _stream(self, name: str):
        for batch in gen_batch(self._produce_examples(name), self._batch_size):
            images, categories = zip(*list(batch))
            yield {'images': images, 'labels': categories}

    def train_stream(self):
        for batch in self._stream('train'):
            yield batch

    def valid_stream(self):
        for batch in self._stream('valid'):
            yield batch

    def predict_stream(self):
        for batch in gen_batch(self._produce_predict_examples(), self._batch_size):
            images, ids = zip(*list(batch))
            yield {'images': images, 'ids': ids}

    @property
    def num_classes(self):
        self._load_meta()
        return len(self._categories.index)

    @property
    def shape(self):
        return [180, 180, 3]

    @property
    def data_root(self):
        return self._data_root

    @property
    def num_batches(self):
        return {'train': round(12371293*(1-self._valid_percent)/self._batch_size),
                'valid': round(12371293*self._valid_percent/self._batch_size)}

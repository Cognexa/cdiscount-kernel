import logging

import os
import os.path as path
import random
import cxflow as cx
import pandas as pd
import io
import bson
import skimage.data
import skimage.io
from itertools import takewhile


def gen_batch(iterable, size):
    source = iter(iterable)
    while True:
        chunk = [val for _, val in zip(range(size), source) if val is not None]
        if not chunk:
            raise StopIteration
        yield chunk


class CDCNaiveDataset(cx.datasets.BaseDataset):

    FILENAMES = ['category_names.7z', 'sample_submission.7z', 'train_example.bson']
    TRAIN_FILE = 'train_example.bson'
    CATEGORIES_FILE = 'categories.csv'
    VISUAL_DIR = 'visual'

    def _configure_dataset(self, data_root: str='data',
                           split_file: str='split.csv',
                           valid_percent: float=0.1,
                           batch_size: int=10,
                           **kwargs):
        self._data_root = data_root
        self._split_file = split_file
        self._valid_percent = valid_percent
        self._batch_size = batch_size
        self._split = None
        self._categories = None

    def validate(self):
        """
        Validate data download.

        TODO: check file hashes
        """
        for filename in self.FILENAMES:
            assert path.exists(path.join(self._data_root, filename)), \
                'Missing `{}` file in `{}`'.format(filename, self._data_root)
        logging.info('All files are in place')

    def visualize(self):
        """
        Save a few images from the TRAIN_FILE.

        For more elaborate visualizations and statistics, see other kernels such as
        <https://www.kaggle.com/bguberfain/just-showing-a-few-images>
        <https://www.kaggle.com/bguberfain/naive-statistics>
        """
        os.makedirs(self.VISUAL_DIR, exist_ok=True)
        data = bson.decode_file_iter(open(path.join(self._data_root, self.TRAIN_FILE), 'rb'))
        for i, example in takewhile(lambda x: x[0] < 10, enumerate(data)):
            for j, image in enumerate(example['imgs']):
                skimage.io.imsave(path.join(self.VISUAL_DIR, 'image_{}_{}.jpg'.format(i, j)),
                                  skimage.data.imread(io.BytesIO(image['picture'])))
        logging.info('Images saved to `%s`', self.VISUAL_DIR)

    def split(self):
        ids = []
        categories = []
        for example in bson.decode_file_iter(open(path.join(self._data_root, self.TRAIN_FILE), 'rb')):
            ids.append(example['_id'])
            categories.append(example['category_id'])
        size = len(ids)
        train_size = int(size*(1-self._valid_percent))
        valid_size = (size-train_size)
        split = ['train']*train_size + ['valid']*valid_size
        random.shuffle(split)
        split_df = pd.DataFrame({'id': ids, 'split': split})
        split_path = path.join(self._data_root, self._split_file)
        split_df.to_csv(split_path, index=False)
        logging.info('Split train-valid of size %s-%s was written to `%s`', train_size, valid_size, split_path)

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

    def _data_iterator(self, name: str):
        self._load_meta()
        for example in bson.decode_file_iter(open(path.join(self._data_root, self.TRAIN_FILE), 'rb')):
            if self._split.loc[example['_id']]['split'] == name:
                for image in example['imgs']:
                    yield (skimage.data.imread(io.BytesIO(image['picture'])), example['category_id'])

    def _stream(self, name: str):
        for batch in gen_batch(self._data_iterator(name), self._batch_size):
            images, categories = zip(*list(batch))
            categories = [self._categories.loc[category]['class'] for category in categories]
            images = [image/255 for image in images]
            yield {'images': images, 'labels': categories}

    def train_stream(self):
        for batch in self._stream('train'):
            yield batch

    def valid_stream(self):
        for batch in self._stream('valid'):
            yield batch

    @property
    def num_classes(self):
        self._load_meta()
        return len(self._categories.index)

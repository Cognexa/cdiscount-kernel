import logging
import cv2
import bson
import os.path as path
import numpy as np
import pandas as pd
from .cdc_dataset import CDCNaiveDataset
from .util import decode_image, gen_batch
import random


class FastDataset(CDCNaiveDataset):
    """
    FastDataset provides fast random access to the train data.

    Images are resized to the specified size and saved as jpegs to a single binary file whilst byte sizes and
    offsets are saved to an extra meta file. This allows to fast seek, read and decode any image.

    This dataset may be used with both CDCNaiveNet and XCeptionNet but first, one need to run
    ```
    cxflow dataset resize cdc/xception.yaml
    ```
    to resize the data and calculate offsets.

    NOTE: this dataset requires OpenCV.
    """

    def _configure_dataset(self, size: int, **kwargs):
        super()._configure_dataset(**kwargs)
        self._meta = None
        self._size = size

    @property
    def thumb_filename(self) -> str:
        return 'images_{}x{}.bin'.format(self._size, self._size)

    @property
    def meta_filename(self) -> str:
        return 'images_{}x{}.csv'.format(self._size, self._size)

    def _load_meta(self):
        for file in [self.thumb_filename, self.meta_filename]:
            if not path.exists(path.join(self._data_root, file)):
                raise FileNotFoundError('File `{}` not found. Run `cxflow dataset resize ...` first.'.format(file))
        super()._load_meta()
        if self._meta is None:
            self._meta = pd.read_csv(path.join(self._data_root, self.meta_filename))

    def resize(self):
        done = 0
        sizes = [0]
        categories = []
        logging.info('Resizing image to [%s, %s], this may take a few hours', self._size, self._size)
        with open(path.join(self._data_root, self.thumb_filename), 'wb') as file_:
            for example in bson.decode_file_iter(open(path.join(self._data_root, self.TRAIN_FILE), 'rb')):
                for image in example['imgs']:
                    if self._size != 180:
                        image = cv2.imdecode(np.frombuffer(image['picture'], dtype=np.uint8), cv2.IMREAD_COLOR)
                        image = cv2.resize(image, (self._size, self._size))
                        _, buff = cv2.imencode('.jpg', image)
                    else:
                        buff = image['picture']
                    sizes.append(sizes[-1] + file_.write(bytearray(buff)))
                    categories.append(example['category_id'])
                    done += 1
                    if done % 100000 == 0:
                        logging.info('%s done', done)
                        logging.info('%s bytes written', sizes[-1])

        # save (category_id -> integer class) mapping
        categories_path = path.join(self._data_root, self.CATEGORIES_FILE)
        logging.info('Saving categories mapping to `{}`'.format(categories_path))
        unique_categories = sorted(list(set(categories)))
        categories_df = pd.DataFrame({'category_id': unique_categories, 'class': list(range(len(unique_categories)))})
        categories_df.to_csv(categories_path, index=False)

        logging.info('Mapping categories to classes to `{}`'.format(categories_path))
        mapping = {}
        for class_, category in enumerate(unique_categories):
            mapping[category] = class_
        classes = [mapping[category] for category in categories]
        offsets_df = pd.DataFrame({'class': classes,
                                   'offset': sizes[:-1],
                                   'size': [to-from_ for to, from_ in zip(sizes[1:], sizes)]})
        offsets_df.to_csv(path.join(self._data_root, self.meta_filename))

    def split(self):
        # generate random split
        self._load_meta()
        size = len(self._meta.index)
        train_size = int(size*(1-self._valid_percent))
        valid_size = (size-train_size)
        split = ['train']*train_size + ['valid']*valid_size
        random.shuffle(split)

        # save split
        split_df = pd.DataFrame({'split': split})
        split_path = path.join(self._data_root, self._split_file)
        split_df.to_csv(split_path)
        logging.info('Split train-valid of size %s-%s was written to `%s`', train_size, valid_size, split_path)

    def _produce_examples(self, name: str):
        self._load_meta()
        examples = zip(self._meta['class'], self._meta['offset'], self._meta['size'], self._split['split'])
        examples = [example[:3] for example in examples if example[3] == name]
        if name == 'train':
            logging.info('Shuffling train examples')
            examples = list(examples)
            random.shuffle(examples)
            logging.info('Shuffled')

        with open(path.join(self._data_root, self.thumb_filename), 'rb') as file_:
            for class_, offset, size in examples:
                file_.seek(offset)
                yield decode_image(file_.read(size))/255, class_

    def _produce_predict_examples(self):
        for example in bson.decode_file_iter(open(path.join(self._data_root, self.TEST_FILE), 'rb')):
            for image in example['imgs']:
                image = cv2.imdecode(np.frombuffer(image['picture'], dtype=np.uint8), cv2.IMREAD_COLOR)
                image = cv2.resize(image, (self._size, self._size))/255
                yield (image, example['_id'])

    def predict_stream(self):
        for batch in gen_batch(self._produce_predict_examples(), self._batch_size):
            images, ids = zip(*list(batch))
            yield {'images': images, 'ids': ids}

    @property
    def shape(self):
        return [self._size, self._size, 3]

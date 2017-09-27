import cxflow as cx
import pandas as pd
import os.path as path
from .cdc_dataset import CDCNaiveDataset
from .util import major_voting
import numpy as np


class CDCPredictionHook(cx.hooks.AccumulateVariables):
    """
    Accumulate ids and predictions, compute major vote and dump the results
    to `submission.csv` which can be directly submitted.
    """

    def __init__(self, dataset: CDCNaiveDataset, **kwargs):
        super().__init__(**kwargs, variables=['ids', 'predictions'])
        self._class_to_category = pd.read_csv(path.join(dataset.data_root, dataset.CATEGORIES_FILE), index_col=1)

    def after_epoch(self, epoch_id, epoch_data):
        # map integer classes back to cdiscount categories
        ids = self._accumulator['predict']['ids']
        predictions = [self._class_to_category.loc[np.argmax(prediction)]['category_id']
                       for prediction in self._accumulator['predict']['predictions']]

        # major voting
        ids, predictions = major_voting(ids, predictions)

        # save the predictions
        submission = pd.DataFrame({'_id': ids, 'category_id': predictions})
        submission.to_csv('submission.csv', index=False)
        super().after_epoch()  # clean the accumulator

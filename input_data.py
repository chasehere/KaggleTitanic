import numpy as np
import pandas as pd
 
class DataSet(object):
  
  def __init__(self,features,targets):
    assert features.shape[0] == targets.shape[0], (
        'features.shape: %s, targets.shape: %s' % 
        (features.shape, targets.shape))

    self._num_examples = features.shape[0]
    self._features = features
    self._targets = targets
    self._epochs_completed = 0
    self._index_in_epochs = 0

    @property
    def features(self):
      return self._features

    @property
    def targets(self):
      return self._targets

    @property
    def num_examples(self):
      return self._num_examples

    @property
    def epochs_completed(self):
      return self._epochs_completed

    def next_batch(self, batch_size):
      """ Return the next 'batch_size' examples from this data set """
      start = self._index_in_epoch
      self._index_in_epoch += batch_size
      if self._index_in_epoch > self._num_examples:
        # Finished epoch
        self._epochs_completed += 1

        # Shuffle the data
        perm numpy.arange(self._num_examples)
        np.random.shuffle(perm)
        self._features = self._features[perm]
        self._targets = self._targets[perm]

        # start next epoch
        start = 0
        self._index_in_epoch = batch_size
        assert batch_size <= self._num_examples

      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]

def read_data_sets():
  class DataSets(object):
    pass

  data_sets = DataSets()

  # load/preprocess the data

  # split the data in test/train/validation sets

  return data_sets

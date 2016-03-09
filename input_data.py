import numpy as np
import pandas as pd

class DataSet(object):
  def __init__(self,ids,features,targets):
    assert features.shape[0] == targets.shape[0], (
        'features.shape: %s, targets.shape: %s' % 
        (features.shape, targets.shape))

    self._num_examples = features.shape[0]
    self._ids = ids
    self._features = features
    self._targets = targets
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def ids(self):
    return self._ids

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
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._features = self._features[perm]
      self._targets = self._targets[perm]

      # start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples

    end = self._index_in_epoch
    return self._features[start:end], self._targets[start:end]


def read_data_sets():
  class DataSets(object):
    pass

  data_sets = DataSets()

  # load/preprocess the data
  train = pd.read_csv('data/train.csv')
  test = pd.read_csv('data/test.csv')

  # only keep a subset of data
  train = train[['PassengerId','Survived','Pclass','Sex','Fare']]
  test = test[['PassengerId','Pclass','Sex','Fare']]

  # replace NaN's with average values
  train = train.fillna(train.mean())
  test = test.fillna(test.mean())

  # cap Fares to 100 to get a better distribution
  train['Fare'][ train['Fare'] > 100 ] = 100.0
  test['Fare'][ test['Fare'] > 100 ] = 100.0

  # convert sex to number
  train['Sex'][ train['Sex']=='female'] = 1.0
  train['Sex'][ train['Sex']=='male'] = 0.0
  test['Sex'][ test['Sex']=='female'] = 1.0
  test['Sex'][ test['Sex']=='male'] = 0.0

  # Normalize 
  train['Pclass'] /= train['Pclass'].max()
  train['Fare'] /= train['Fare'].max()
  test['Pclass'] /= test['Pclass'].max()
  test['Fare'] /= test['Fare'].max()
 
  # dummy target for test set
  test['Survived'] = 0.0

  # split the data in test/train/validation sets
  VALIDATION_SIZE = 267

  train_ids = train[VALIDATION_SIZE:][['PassengerId']]
  train_features = train[VALIDATION_SIZE:].drop(['PassengerId','Survived'],axis=1)
  train_labels = train[VALIDATION_SIZE:][['Survived']]
  valid_ids = train[:VALIDATION_SIZE][['PassengerId']]
  valid_features = train[:VALIDATION_SIZE].drop(['PassengerId','Survived'],axis=1)
  valid_labels = train[:VALIDATION_SIZE][['Survived']]
  test_ids = np.array( test[['PassengerId']] )
  test_features = np.array( test.drop(['PassengerId','Survived'], axis=1) )
  test_labels = np.array( test[['Survived']] )

  data_sets.train = DataSet(np.array(train_ids),np.array(train_features), np.array(train_labels))
  data_sets.validation = DataSet(np.array(valid_ids),np.array(valid_features), np.array(valid_labels))
  data_sets.test = DataSet(test_ids,test_features, test_labels)

  return data_sets

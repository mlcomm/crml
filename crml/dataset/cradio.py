from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import pickle

from tensorflow.contrib.learn.python.learn.datasets import base

class DataSet(object):
    def __init__(self, 
                 data, 
                 labels,
                 seed=None):
        self._data = data
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        
        assert data.shape[0] == labels.shape[0], (
            'data.shape: %s labels.shape: %s' %(data.shape, labels.shape))
        self._num_examples = data.shape[0]
    
    @property
    def data(self):
        return self._data
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self._data = self.data[perm0]
            self._labels = self.labels[perm0]
        
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            #Finish epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            data_rest_part = self._data[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            #Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._data = self.data[perm]
                self._labels = self.labels[perm]
                
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            data_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return numpy.concatenate(
              (data_rest_part, data_new_part), axis=0), numpy.concatenate(
                  (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end], self._labels[start:end]
        
    
def read_data_sets(data_path,
                  test_size=5000,
                  seed=None):
    
    with open(data_path,'rb') as f:
        [data, labels, _,]= pickle.load(f)
    
    if not 0 <= test_size <= len(data):
        raise ValueError('Test size should be between 0 and {}. Received: {}.'
                     .format(len(data), test_size))
        
    train_data = data[0:test_size]
    train_labels = labels[0:test_size]
    test_data = data[test_size:]
    test_labels = labels[test_size:]
    
    options = dict(seed=seed)
    
    train = DataSet(train_data, train_labels, **options)
    test = DataSet(test_data, test_labels, **options)
    
    return base.Datasets(train=train, validation = None, test=test)

# imports
from data_utils import get_token_ids
from data_utils import get_model_input
from data_utils import inputs_to_batch
from model import load_model

import os
import logging
from six.moves import xrange

import numpy as np
import tensorflow as tf


# classes
class Evaluater:
    def __init__(self, vocabulary, checkpoint_path=None, model=None,
                 session=None):
        if model is None:
            model = load_model()

        if session is None:
            logging.debug('creating session')
            init = tf.global_variables_initializer()

            sess = tf.Session()
            sess.run(init)

            checkpoint_file_path = self._get_checkpoint_file_path(
                checkpoint_path)

            model.load_checkpoint(sess, checkpoint_file_path)

            self.session = sess
        else:
            self.session = session

        self.model = model

        self.vocabulary = vocabulary

    def _get_checkpoint_file_path(self, path):
        if os.path.isdir(path):
            files = os.listdir(path)
            checkpoints = list(set([file.split('.')[0]
                                    for file in files]))
            latest = sorted(checkpoints)[-1]

            return './' + os.path.join(path, latest)
        else:
            return path

    def _get_input(self, text):
        input_token_ids = get_token_ids(text, self.vocabulary)
        output_token_ids = get_token_ids('hello', self.vocabulary)

        bucket_id, model_input = get_model_input(self.model.buckets,
                                                 input_token_ids,
                                                 output_token_ids)

        return bucket_id, model_input

    def _get_inputs(self, inp, is_batch):
        bucket_id = None
        inputs_batch = []
        outputs_batch = []
        if is_batch:
            for text in sorted(inp, key=len):
                bucket_id, model_input = self._get_input(text)

                bucket_id = bucket_id

                inputs_batch.append(model_input[0])
                outputs_batch.append(model_input[1])
        else:
            bucket_id, model_input = self._get_input(inp)
            for _ in xrange(self.model.batch_size):
                inputs_batch.append(model_input[0])
                outputs_batch.append(model_input[1])

            bucket_id = bucket_id

        inputs_batch = inputs_to_batch(inputs_batch, len(inputs_batch[0]),
                                       self.model.batch_size)
        outputs_batch = inputs_to_batch(outputs_batch, len(outputs_batch[0]),
                                        self.model.batch_size)
        target_weights = [np.ones(self.model.batch_size)
                          for _ in xrange(len(outputs_batch))]

        return inputs_batch, outputs_batch, target_weights, bucket_id

    def close(self):
        self.session.close()

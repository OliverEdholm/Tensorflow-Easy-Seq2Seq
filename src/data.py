from data_utils import _START_VOCAB
from data_utils import UNK_ID
from data_utils import PAD_ID
from data_utils import get_model_input
from data_utils import inputs_to_batch
from data_utils import split_sentence_tokenizer
from data_utils import get_token_ids
from data_utils import get_batch_target_weights
from utils import get_sorted_buckets
from utils import get_text_file

import logging
from collections import Counter
from random import choice
from random import randint
from six.moves import xrange
from six.moves import cPickle

import numpy as np
from tqdm import tqdm


# classes
class Vocabulary:
    def __init__(self, data_points_path, max_vocabulary_size, tokenizer):
        self.data_points_path = data_points_path
        self.max_vocabulary_size = max_vocabulary_size
        self.tokenizer = tokenizer

        self.token_to_id, self.id_to_token = self._get_vocabulary()

        self.size = len(self.token_to_id.keys())

    def _normalize_text(self, text):
        return text.lower()

    def _get_vocabulary(self):
        data_points = get_text_file(self.data_points_path)

        all_tokens = []
        for data_point in tqdm(data_points):
            normalized_data_point = self._normalize_text(data_point)

            tokens = self.tokenizer(normalized_data_point)
            all_tokens.extend(tokens)

        tokens_frequencies = Counter(all_tokens)

        vocabulary_tokens = tokens_frequencies.most_common(
            self.max_vocabulary_size - len(_START_VOCAB))
        vocabulary_tokens = list(map(lambda x: x[0], vocabulary_tokens))
        vocabulary_tokens = _START_VOCAB + vocabulary_tokens

        token_to_id = {token: idx
                       for idx, token in enumerate(vocabulary_tokens)}
        id_to_token = {idx: token
                       for idx, token in enumerate(vocabulary_tokens)}

        return (token_to_id, id_to_token)

    def get_token(self, token_id):
        return self.id_to_token[token_id]

    def get_id(self, token):
        return self.token_to_id.get(token, UNK_ID)


class DataSet:
    def __init__(self, inputs_path, outputs_path, input_vocab_size,
                 output_vocab_size, buckets, eval_data_percentage=0.05,
                 tokenizer=split_sentence_tokenizer):
        logging.info('creating dataset')
        buckets = get_sorted_buckets(buckets)

        self.buckets = buckets
        self.tokenizer = tokenizer
        self.eval_data_percentage = eval_data_percentage

        logging.debug('making input vocabulary')
        self.input_vocabulary = Vocabulary(inputs_path, input_vocab_size,
                                           tokenizer)
        logging.debug('making output vocabulary')
        self.output_vocabulary = Vocabulary(outputs_path, output_vocab_size,
                                            tokenizer)

        self.input_vocabulary_size = self.input_vocabulary.size
        self.output_vocabulary_size = self.output_vocabulary.size

        logging.debug('creating dataset')
        self.train_data_set, self.eval_data_set = self._get_data_sets(
            inputs_path, outputs_path)

    def _get_tokens(self, token_ids, vocabulary):
        return [vocabulary.get_token(token_id)
                for token_id in token_ids]

    def _indices_to_values(self, indices, encoder_inputs, decoder_inputs):
        new_encoder_inputs = []
        new_decoder_inputs = []

        for idx in indices:
            new_encoder_inputs.append(encoder_inputs[idx])
            new_decoder_inputs.append(decoder_inputs[idx])

        return new_encoder_inputs, new_decoder_inputs

    def _distribute_data(self, encoder_inputs, decoder_inputs):
        train_indices = [i
                         for i in xrange(len(encoder_inputs))]

        eval_size = round(len(encoder_inputs) * self.eval_data_percentage)

        eval_indices = []
        for _ in xrange(eval_size):
            idx = randint(0, len(train_indices) - 1)

            eval_indices.append(train_indices[idx])
            train_indices.pop(idx)

        train_encoder_inputs, train_decoder_inputs = self._indices_to_values(
            train_indices, encoder_inputs, decoder_inputs)
        eval_encoder_inputs, eval_decoder_inputs = self._indices_to_values(
            eval_indices, encoder_inputs, decoder_inputs)

        return train_encoder_inputs, train_decoder_inputs, \
            eval_encoder_inputs, eval_decoder_inputs

    def _get_data_set(self, inputs, outputs):
        data_set = [[] for _ in xrange(len(self.buckets))]

        for inp, out in tqdm(zip(inputs, outputs), total=len(inputs)):
            input_token_ids = get_token_ids(inp, self.input_vocabulary,
                                            self.tokenizer)
            output_token_ids = get_token_ids(out, self.output_vocabulary,
                                             self.tokenizer)

            bucket_id, model_input = get_model_input(self.buckets,
                                                     input_token_ids,
                                                     output_token_ids)

            if bucket_id is not None and model_input is not None:
                data_set[bucket_id].append(model_input)

        return data_set

    def _get_data_sets(self, inputs_path, outputs_path):
        inputs = get_text_file(inputs_path)
        outputs = get_text_file(outputs_path)

        train_inputs, train_outputs, \
            eval_inputs, eval_outputs = self._distribute_data(inputs, outputs)

        logging.debug('getting traning dataset')
        train_data_set = self._get_data_set(train_inputs, train_outputs)
        logging.debug('getting evaluation dataset')
        eval_data_set = self._get_data_set(eval_inputs, eval_outputs)

        return train_data_set, eval_data_set

    def _get_correct_data_set(self, use_train_data):
        if use_train_data:
            return self.train_data_set
        else:
            return self.eval_data_set

    def _get_random_inputs(self, n_inputs, bucket_id, use_train_data):
        data_set = self._get_correct_data_set(use_train_data)

        encoder_inputs = []
        decoder_inputs = []
        for _ in xrange(n_inputs):
            encoder_input, decoder_input = choice(data_set[bucket_id])

            encoder_inputs.append(encoder_input)
            decoder_inputs.append(decoder_input)

        return encoder_inputs, decoder_inputs

    def _inputs_to_batch(self, inputs, inputs_size, batch_size):
        batch = []
        for token_idx in xrange(inputs_size):
            batch_row = [inputs[input_idx][token_idx]
                         for input_idx in xrange(batch_size)]
            batch_row = np.array(batch_row, dtype=np.int32)

            batch.append(batch_row)

        return batch

    def save(self, file_path):
        logging.info('saving dataset at {}'.format(file_path))
        with open(file_path, 'wb') as pkl_file:
            pkl_file.flush()

            cPickle.dump(self, pkl_file)

    @staticmethod
    def load(file_path):
        logging.info('loading dataset at {}'.format(file_path))
        with open(file_path, 'rb') as pkl_file:
            return cPickle.load(pkl_file)

    def get_random_batch(self, batch_size, bucket_id, use_train_data=True):
        encoder_inputs, decoder_inputs = self._get_random_inputs(
            batch_size, bucket_id, use_train_data)

        encoder_size, decoder_size = self.buckets[bucket_id]
        batch_encoder_inputs = inputs_to_batch(encoder_inputs,
                                               encoder_size, batch_size)
        batch_decoder_inputs = inputs_to_batch(decoder_inputs,
                                               decoder_size, batch_size)
        batch_target_weights = get_batch_target_weights(decoder_inputs,
                                                        decoder_size,
                                                        batch_size)

        return batch_encoder_inputs, batch_decoder_inputs, batch_target_weights

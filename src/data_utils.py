# imports
import re
import logging
from six.moves import xrange

import numpy as np
import tensorflow as tf


# variables
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


# functions
def get_batch_target_weights(decoder_inputs, decoder_size, batch_size):
    batch_weights = []

    for token_idx in xrange(decoder_size):
        batch_weight = np.ones(batch_size, dtype=np.float32)

        for batch_idx in xrange(batch_size):
            if token_idx < decoder_size - 1:
                target = decoder_inputs[batch_idx][token_idx + 1]

            if token_idx == decoder_size - 1 or target == PAD_ID:
                batch_weight[batch_idx] = 0.0

        batch_weights.append(batch_weight)

    return batch_weights


def split_sentence_tokenizer(sentence):
    sentence = tf.compat.as_bytes(sentence)  # is needed to feed it to regex

    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(re.compile(b"([.,!?\"':;)(])"),
                     space_separated_fragment))

    return [w
            for w in words
            if w]


def add_go_and_eos(token_ids):
    token_ids = [GO_ID] + token_ids + [EOS_ID]

    return token_ids


def get_bucket_id(buckets, input_token_ids, output_token_ids):
    inp_size = len(input_token_ids)
    out_size = len(output_token_ids)
    for bucket_id, bucket in enumerate(buckets):
        bucket_inp_size, bucket_out_size = bucket

        if inp_size <= bucket_inp_size and out_size <= bucket_out_size:
            return bucket_id
    else:
        return None


def pad_token_ids(token_ids, length):
    n_pads = length - len(token_ids)
    for _ in xrange(n_pads):
        token_ids.append(PAD_ID)

    return token_ids


def get_model_input(buckets, input_token_ids, output_token_ids):
    output_token_ids = add_go_and_eos(output_token_ids)

    bucket_id = get_bucket_id(buckets, input_token_ids, output_token_ids)

    if bucket_id is None:
        logging.warning('found data that couldn\'t fit any bucket, '
                        'ignoring it')
        return None, None

    bucket_inp_size, bucket_out_size = buckets[bucket_id]

    input_token_ids = pad_token_ids(input_token_ids,
                                    bucket_inp_size)
    output_token_ids = pad_token_ids(output_token_ids,
                                     bucket_out_size)

    input_token_ids = input_token_ids[::-1]  # for better performance
    model_input = (np.array(input_token_ids), np.array(output_token_ids))

    return bucket_id, model_input


def get_token_ids(text, vocabulary, tokenizer=split_sentence_tokenizer):
    tokens = tokenizer(text)

    return [vocabulary.get_id(token)
            for token in tokens]


def inputs_to_batch(inputs, inputs_size, batch_size):
    batch = []
    for token_idx in xrange(inputs_size):
        batch_row = [inputs[input_idx][token_idx]
                     for input_idx in xrange(batch_size)]
        batch_row = np.array(batch_row, dtype=np.int32)

        batch.append(batch_row)

    return batch

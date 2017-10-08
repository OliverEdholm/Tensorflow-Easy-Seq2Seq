# imports
from config import PARAM_FILE_PATH
from config import DEFAULT_BUCKETS_STRING
from config import DEFAULT_LAYER_SIZE
from config import DEFAULT_N_LAYERS
from config import DEFAULT_MAX_GRADIENT_NORM
from config import DEFAULT_BATCH_SIZE
from config import DEFAULT_LEARNING_RATE
from config import DEFAULT_LEARNING_RATE_DECAY_FACTOR
from config import DEFAULT_RNN_CELL
from config import DEFAULT_N_SAMPLES
from config import DEFAULT_FORWARD_ONLY
from model_utils import get_rnn_cell
from utils import get_sorted_buckets
from utils import get_buckets

import logging
from six.moves import xrange
from six.moves import cPickle

import numpy as np
import tensorflow as tf

# variables
EMBEDDING_LAYER_NAME = "<tf.Variable 'embedding_attention_seq2seq/rnn/embedding_wrapper/embedding:0' shape=(20000, 256) dtype=float32_ref>"


# classes
class Seq2Seq:
    def __init__(self,
                 input_vocab_size,
                 output_vocab_size,
                 buckets=get_buckets(DEFAULT_BUCKETS_STRING),
                 layer_size=DEFAULT_LAYER_SIZE,
                 n_layers=DEFAULT_N_LAYERS,
                 max_gradient_norm=DEFAULT_MAX_GRADIENT_NORM,
                 batch_size=DEFAULT_BATCH_SIZE,
                 learning_rate=DEFAULT_LEARNING_RATE,
                 learning_rate_decay_factor=DEFAULT_LEARNING_RATE_DECAY_FACTOR,
                 rnn_cell=get_rnn_cell(DEFAULT_RNN_CELL),
                 n_samples=DEFAULT_N_SAMPLES,
                 forward_only=DEFAULT_FORWARD_ONLY):
        logging.info('initializing Seq2Seq model')
        buckets = get_sorted_buckets(buckets)

        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.buckets = buckets
        self.layer_size = layer_size
        self.n_layers = n_layers
        self.max_gradient_norm = max_gradient_norm
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.rnn_cell = rnn_cell
        self.n_samples = n_samples
        self.forward_only = forward_only

        logging.debug('saving params')
        self._save_params()

        logging.debug('assigning learning rate')
        self.learning_rate = tf.Variable(float(self.learning_rate),
                                         trainable=False)
        self.learning_rate_decay_operation = self.learning_rate.assign(
                        self.learning_rate * self.learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        logging.debug('creating placeholders')
        self.encoder_inputs = [self._get_val_placeholder('encoder', i)
                               for i in xrange(buckets[-1][0])]

        self.decoder_inputs = [self._get_val_placeholder('decoder', i)
                               for i in xrange(buckets[-1][1] + 1)]
        self.target_weights = [self._get_val_placeholder('weight', i,
                                                         dtype=tf.float32)
                               for i in xrange(buckets[-1][1] + 1)]

        self.outputs, self.losses = self._get_model_with_buckets()

        logging.debug('building saver')
        self.saver = tf.train.Saver(tf.global_variables())

        if not forward_only:
            out = self._get_gradient_norms_and_updates()
            self.gradient_norms, self.updates = out

        self.embedding_operation = self._get_embedding_operation()

    def _save_params(self):
        params = {
            'input_vocab_size': self.input_vocab_size,
            'output_vocab_size': self.output_vocab_size,
            'buckets': self.buckets,
            'layer_size': self.layer_size,
            'n_layers': self.n_layers,
            'max_gradient_norm': self.max_gradient_norm,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'learning_rate_decay_factor': self.learning_rate_decay_factor,
            'rnn_cell': self.rnn_cell,
            'n_samples': self.n_samples,
            'forward_only': self.forward_only
        }

        with open(PARAM_FILE_PATH, 'wb') as pkl_file:
            cPickle.dump(params, pkl_file)

    def _get_embedding_operation(self):
        return tf.trainable_variables()[2]

    def _get_softmax_loss_func_and_output_proj(self):
        logging.debug('getting softmax loss function')
        use_sampled_softmax = self.n_samples > 0 and \
            self.n_samples < self.output_vocab_size
        if use_sampled_softmax:
            w = tf.get_variable('proj_w',
                                [self.layer_size, self.output_vocab_size],
                                dtype=tf.float32)
            w_t = tf.transpose(w)
            b = tf.get_variable('proj_b',
                                [self.output_vocab_size],
                                dtype=tf.float32)

            def get_sampled_loss(labels, logits):
                labels = tf.reshape(labels, [-1, 1])

                return tf.nn.sampled_softmax_loss(
                    weights=w_t, biases=b, labels=labels, inputs=logits,
                    num_sampled=self.n_samples,
                    num_classes=self.output_vocab_size)

            softmax_loss_function = get_sampled_loss
            output_projection = (w, b)
        else:
            softmax_loss_function = None
            output_projection = None

        return softmax_loss_function, output_projection

    def _get_cell(self):
        logging.debug('getting rnn cell')
        single_cell = self.rnn_cell(self.layer_size)

        if self.n_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([single_cell] * self.n_layers)
        else:
            cell = single_cell

        return cell

    def _get_val_placeholder(self, name, idx, dtype=tf.int32):
        return tf.placeholder(dtype, shape=[None], name='{}_{}'.format(name,
                                                                       idx))

    def _get_model_with_buckets(self):
        logging.debug('getting model with buckets')
        targets = [self.decoder_inputs[i + 1]
                   for i in xrange(len(self.decoder_inputs) - 1)]

        out = self._get_softmax_loss_func_and_output_proj()
        softmax_loss_function, output_projection = out

        cell = self._get_cell()

        def seq2seq_func(encoder_inputs, decoder_inputs, do_decode):
            logging.debug('building model with bucket ({}, {})'.format(
                          len(encoder_inputs), len(decoder_inputs)))
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
              encoder_inputs, decoder_inputs, cell,
              num_encoder_symbols=self.input_vocab_size,
              num_decoder_symbols=self.output_vocab_size,
              embedding_size=self.layer_size,
              output_projection=output_projection,
              feed_previous=do_decode)

        logging.debug('building models')
        outputs, losses = tf.contrib.legacy_seq2seq.model_with_buckets(
           self.encoder_inputs, self.decoder_inputs, targets,
           self.target_weights, self.buckets,
           lambda x, y: seq2seq_func(x, y, self.forward_only),
           softmax_loss_function=softmax_loss_function)

        if self.forward_only:
            if output_projection is not None:
                logging.debug('fixing forward only')
                for i in xrange(len(self.buckets)):
                    w = output_projection[0]
                    b = output_projection[1]
                    new_bucket_outputs = [tf.matmul(output, w) + b
                                          for output in outputs[i]]

                    outputs[i] = new_bucket_outputs

        return outputs, losses

    def _get_gradient_norms_and_updates(self):
        logging.debug('getting gradient norms and updates')
        params = tf.trainable_variables()

        gradient_norms = []
        updates = []
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        for i in xrange(len(self.buckets)):
            logging.debug('getting from bucket ({}, {})'.format(
                          *self.buckets[i]))
            gradients = tf.gradients(self.losses[i], params)
            clipped_gradients, norm = tf.clip_by_global_norm(
                gradients, self.max_gradient_norm)
            gradient_norms.append(norm)
            updates.append(optimizer.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step))

        return gradient_norms, updates

    def step(self, session, encoder_inputs, decoder_inputs, target_weights,
             bucket_id, forward_only, get_embedding=False):
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError('Encoder length must be equal to one in bucket.')
        elif len(decoder_inputs) != decoder_size:
            raise ValueError('Decoder length must be equal to one in bucket.')
        elif len(target_weights) != decoder_size:
            raise ValueError('Weights length must be equal to one in bucket.')

        input_feed = {}
        for i in xrange(encoder_size):
            input_feed[self.encoder_inputs[i].name] = encoder_inputs[i]
        for i in xrange(decoder_size):
            input_feed[self.decoder_inputs[i].name] = decoder_inputs[i]
            input_feed[self.target_weights[i].name] = target_weights[i]

        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        if forward_only:
            output_feed = [self.losses[bucket_id]]

            for i in xrange(decoder_size):
                output_feed.append(self.outputs[bucket_id][i])
        else:
            output_feed = [self.updates[bucket_id],
                           self.gradient_norms[bucket_id],
                           self.losses[bucket_id]]

        if get_embedding:
            output_feed.append(self.embedding_operation)

        outputs = session.run(output_feed, input_feed)

        if not forward_only:
            out = [outputs[1], outputs[2], None]
        else:
            out = [None, outputs[0], outputs[1:]]

        if get_embedding:
            out.append(outputs[-2])

        return out

    def save_checkpoint(self, session, checkpoint_path):
        self.saver.save(session, checkpoint_path, global_step=self.global_step)
        logging.debug('saved checkpoint at {}'.format(checkpoint_path))

    def load_checkpoint(self, session, checkpoint_path):
        self.saver.restore(session, checkpoint_path)
        logging.debug('restored checkpoint at {}'.format(checkpoint_path))


def load_model(params_path=PARAM_FILE_PATH, forward_only=False):
    logging.debug('loading model at {}'.format(params_path))
    with open(params_path, 'rb') as pkl_file:
        params = cPickle.load(pkl_file)

        model = Seq2Seq(
            input_vocab_size=params['input_vocab_size'],
            output_vocab_size=params['output_vocab_size'],
            buckets=params['buckets'],
            layer_size=params['layer_size'],
            n_layers=params['n_layers'],
            max_gradient_norm=params['max_gradient_norm'],
            batch_size=params['batch_size'],
            learning_rate=params['learning_rate'],
            learning_rate_decay_factor=params['learning_rate_decay_factor'],
            rnn_cell=params['rnn_cell'],
            n_samples=params['n_samples'],
            forward_only=forward_only
        )

    return model

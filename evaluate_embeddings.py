# imports
import sys
sys.path.append('src')

from config import DEFAULT_CHECKPOINT_PATH
from config import DEFAULT_DATA_SET_PATH
from utils import get_text_file
from data import DataSet
from embedding import Embedder

import logging

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

# setup
logging.basicConfig(level=logging.DEBUG)

# variables
flags = tf.app.flags
flags.DEFINE_string('inputs_file_path', None,
                    'Path to file with inputs to embed.')
flags.DEFINE_string('dim_reduction_method', 'TSNE',
                    'Method to do dimensionality reduction with, either PCA'
                    ' or TSNE.')
flags.DEFINE_string('checkpoint_path', DEFAULT_CHECKPOINT_PATH,
                    'Path to checkpoint, if no file is given it tries to get '
                    'the latest checkpoint in that folder.')
flags.DEFINE_string('data_set_path', DEFAULT_DATA_SET_PATH,
                    'Path to saving destination of dataset.')
FLAGS = flags.FLAGS


# functions
def do_pca(data, n_dimensions, method):
    method = method.lower()
    if method == 'tsne':
        model = PCA(n_components=n_dimensions)
    elif method == 'pca':
        model = TSNE(n_components=n_dimensions)
    else:
        raise ValueError('Dimensionality reduction method {} isn\'t '
                         'implemented or is misspelled.')

    return model.fit_transform(data)


def main():
    data_set = DataSet.load(FLAGS.data_set_path)
    embedder = Embedder(data_set.input_vocabulary, FLAGS.checkpoint_path)

    texts = [text.replace('\n', '')
             for text in get_text_file(FLAGS.inputs_file_path)]

    logging.info('getting embeddings')
    embeddings = [embedder.get_embedding(text)
                  for text in tqdm(texts)]
    embedder.close()

    logging.info('applying dimensionality reduction')
    embeddings_reduced = do_pca(embeddings, 2,
                                FLAGS.dim_reduction_method)

    logging.info('plotting')
    plt.scatter(embeddings_reduced[:, 0], embeddings_reduced[:, 1])
    for text, x_coord, y_coord in zip(texts, embeddings_reduced[:, 0],
                                      embeddings_reduced[:, 1]):
        plt.annotate(text, xy=(x_coord, y_coord), xytext=(0, 0),
                     textcoords='offset points')

    plt.show()


if __name__ == '__main__':
    main()

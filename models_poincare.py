#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Jayant Jain <jayantjain1992@gmail.com>
# Copyright (C) 2017 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Python implementation of Poincaré Embeddings.

These embeddings are better at capturing latent hierarchical information than traditional Euclidean embeddings.
The method is described in detail in `Maximilian Nickel, Douwe Kiela -
"Poincaré Embeddings for Learning Hierarchical Representations" <https://arxiv.org/abs/1705.08039>`_.

The main use-case is to automatically learn hierarchical representations of nodes from a tree-like structure,
such as a Directed Acyclic Graph (DAG), using a transitive closure of the relations. Representations of nodes in a
symmetric graph can also be learned.

This module allows training Poincaré Embeddings from a training file containing relations of graph in a
csv-like format, or from a Python iterable of relations.
/

Examples
--------
Initialize and train a model from a list

.. sourcecode:: pycon

    >>> from gensim.models.poincare import PoincareModel
    >>> relations = [('kangaroo', 'marsupial'), ('kangaroo', 'mammal'), ('gib', 'cat')]
    >>> model = PoincareModel(relations)
    >>> model.train(epochs=50)

Initialize and train a model from a file containing one relation per line

.. sourcecode:: pycon

    >>> from gensim.models.poincare import PoincareModel, PoincareRelations
    >>> from gensim.test.utils import datapath
    >>> file_path = datapath('poincare_hypernyms.tsv')
    >>> model = PoincareModel(PoincareRelations(file_path))
    >>> model.train(epochs=50)

"""

import csv
import logging
from numbers import Integral
import sys
import time
from random import sample

import numpy as np
from collections import defaultdict, Counter
from numpy import random as np_random
from scipy.stats import spearmanr
from six import string_types
from six.moves import zip, range
import math

from gensim import utils, matutils
from gensim.models.keyedvectors import Vocab, BaseKeyedVectors
from gensim.models.utils_any2vec import _save_word2vec_format, _load_word2vec_format
from numpy import float32 as REAL
from joblib import Parallel, delayed

try:
    from autograd import grad  # Only required for optionally verifying gradients while training
    from autograd import numpy as grad_np
    AUTOGRAD_PRESENT = True
except ImportError:
    AUTOGRAD_PRESENT = False

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PoincareModel(utils.SaveLoad):
    """Train, use and evaluate Poincare Embeddings.

    The model can be stored/loaded via its :meth:`~gensim.models.poincare.PoincareModel.save`
    and :meth:`~gensim.models.poincare.PoincareModel.load` methods, or stored/loaded in the word2vec format
    via `model.kv.save_word2vec_format` and :meth:`~gensim.models.poincare.PoincareKeyedVectors.load_word2vec_format`.

    Notes
    -----
    Training cannot be resumed from a model loaded via `load_word2vec_format`, if you wish to train further,
    use :meth:`~gensim.models.poincare.PoincareModel.save` and :meth:`~gensim.models.poincare.PoincareModel.load`
    methods instead.

    An important attribute (that provides a lot of additional functionality when directly accessed) are the
    keyed vectors:

    self.kv : :class:`~gensim.models.poincare.PoincareKeyedVectors`
        This object essentially contains the mapping between nodes and embeddings, as well the vocabulary of the model
        (set of unique nodes seen by the model). After training, it can be used to perform operations on the vectors
        such as vector lookup, distance and similarity calculations etc.
        See the documentation of its class for usage examples.

    """
    def __init__(self, train_data, size=50, alpha=0.1, workers=1, epsilon=1e-5, regularization_coeff=0.0,
                 burn_in=10, burn_in_alpha=0.01, init_range=(-0.001, 0.001), dtype=np.float64, seed=6, root=None):
        """Initialize and train a Poincare embedding model from an iterable of relations.

        Parameters
        ----------
        train_data : {iterable of (str, str), :class:`gensim.models.poincare.PoincareRelations`}
            Iterable of relations, e.g. a list of tuples, or a :class:`gensim.models.poincare.PoincareRelations`
            instance streaming from a file. Note that the relations are treated as ordered pairs,
            i.e. a relation (a, b) does not imply the opposite relation (b, a). In case the relations are symmetric,
            the data should contain both relations (a, b) and (b, a).
        size : int, optional
            Number of dimensions of the trained model.
        alpha : float, optional
            Learning rate for training.
        workers : int, optional
            Number of threads to use for training the model.
        epsilon : float, optional
            Constant used for clipping embeddings below a norm of one.
        regularization_coeff : float, optional
            Coefficient used for l2-regularization while training (0 effectively disables regularization).
        burn_in : int, optional
            Number of epochs to use for burn-in initialization (0 means no burn-in).
        burn_in_alpha : float, optional
            Learning rate for burn-in initialization, ignored if `burn_in` is 0.
        init_range : 2-tuple (float, float)
            Range within which the vectors are randomly initialized.
        dtype : numpy.dtype
            The numpy dtype to use for the vectors in the model (numpy.float64, numpy.float32 etc).
            Using lower precision floats may be useful in increasing training speed and reducing memory usage.
        seed : int, optional
            Seed for random to ensure reproducibility.

        Examples
        --------
        Initialize a model from a list:

        .. sourcecode:: pycon

            >>> from gensim.models.poincare import PoincareModel
            >>> relations = [('kangaroo', 'marsupial'), ('kangaroo', 'mammal'), ('gib', 'cat')]
            >>> model = PoincareModel(relations)

        Initialize a model from a file containing one relation per line:

        .. sourcecode:: pycon

            >>> from gensim.models.poincare import PoincareModel, PoincareRelations
            >>> from gensim.test.utils import datapath
            >>> file_path = datapath('poincare_hypernyms.tsv')
            >>> model = PoincareModel(PoincareRelations(file_path))

        See :class:`~gensim.models.poincare.PoincareRelations` for more options.

        """
        self.train_data = train_data
        self.kv = PoincareKeyedVectors(size)
        self.all_relations = []
        self.node_relations = defaultdict(set)
        self.size = size
        self.train_alpha = alpha  # Learning rate for training
        self.burn_in_alpha = burn_in_alpha  # Learning rate for burn-in
        self.alpha = alpha  # Current learning rate
        self.root = root
        self.workers = workers
        self.epsilon = epsilon
        self.regularization_coeff = regularization_coeff
        self.burn_in = burn_in
        self._burn_in_done = False
        self.dtype = dtype
        self.seed = seed
        self._np_random = np_random.RandomState(seed)
        self.init_range = init_range
        self._loss_grad = None
        self.build_vocab(train_data)

    def build_vocab(self, relations, update=False):
        """Build the model's vocabulary from known relations.

        Parameters
        ----------
        relations : {iterable of (str, str), :class:`gensim.models.poincare.PoincareRelations`}
            Iterable of relations, e.g. a list of tuples, or a :class:`gensim.models.poincare.PoincareRelations`
            instance streaming from a file. Note that the relations are treated as ordered pairs,
            i.e. a relation (a, b) does not imply the opposite relation (b, a). In case the relations are symmetric,
            the data should contain both relations (a, b) and (b, a).
        update : bool, optional
            If true, only new nodes's embeddings are initialized.
            Use this when the model already has an existing vocabulary and you want to update it.
            If false, all node's embeddings are initialized.
            Use this when you're creating a new vocabulary from scratch.

        Examples
        --------
        Train a model and update vocab for online training:

        .. sourcecode:: pycon

            >>> from gensim.models.poincare import PoincareModel
            >>>
            >>> # train a new model from initial data
            >>> initial_relations = [('kangaroo', 'marsupial'), ('kangaroo', 'mammal')]
            >>> model = PoincareModel(initial_relations)
            >>> model.train(epochs=50)
            >>>
            >>> # online training: update the vocabulary and continue training
            >>> online_relations = [('striped_skunk', 'mammal')]
            >>> model.build_vocab(online_relations, update=True)
            >>> model.train(epochs=50)

        """
        old_index2word_len = len(self.kv.index2word)

        logger.info("loading relations from train data..")
        weights = dict()
        for relation in relations:
            for item in relation[:-1]:
                if item in self.kv.vocab:
                    self.kv.vocab[item].count += 1
                else:
                    self.kv.vocab[item] = Vocab(count=1, index=len(self.kv.index2word))
                    self.kv.index2word.append(item)
            node_1, node_2, weight = relation
            node_1_index, node_2_index = self.kv.vocab[node_1].index, self.kv.vocab[node_2].index
            self.node_relations[node_1_index].add(node_2_index)
            relation = (node_1_index, node_2_index)
            weights[relation] = float(weight)
            self.all_relations.append(relation)
        logger.info("loaded %d relations from train data, %d nodes", len(self.all_relations), len(self.kv.vocab))
        self.indices_set = set(range(len(self.kv.index2word)))  # Set of all node indices
        self.indices_array = np.fromiter(range(len(self.kv.index2word)), dtype=int)  # Numpy array of all node indices
        for i in self.indices_set:
            weights[(i, i)] = 0
        self.weights = weights

        self._init_node_probabilities()

        if not update:
            self._init_embeddings()
        else:
            self._update_embeddings(old_index2word_len)

    def _init_embeddings(self):
        """Randomly initialize vectors for the items in the vocab."""
        shape = (len(self.kv.index2word), self.size)
        self.kv.syn0 = self._np_random.uniform(self.init_range[0], self.init_range[1], shape).astype(self.dtype)
        if not (self.root == None):
            self.kv.syn0[self.kv.index2word.index(self.root)] = 0

    def _update_embeddings(self, old_index2word_len):
        """Randomly initialize vectors for the items in the additional vocab."""
        shape = (len(self.kv.index2word) - old_index2word_len, self.size)
        v = self._np_random.uniform(self.init_range[0], self.init_range[1], shape).astype(self.dtype)
        self.kv.syn0 = np.concatenate([self.kv.syn0, v])

    def _init_node_probabilities(self):
        """Initialize a-priori probabilities."""
        counts = np.fromiter((
                self.kv.vocab[self.kv.index2word[i]].count
                for i in range(len(self.kv.index2word))
            ),
            dtype=np.float64, count=len(self.kv.index2word))
        self._node_counts_cumsum = np.cumsum(counts)
        self._node_probabilities = counts / counts.sum()

    @staticmethod
    def _clip_vectors(vectors, epsilon):
        """Clip vectors to have a norm of less than one.

        Parameters
        ----------
        vectors : numpy.array
            Can be 1-D, or 2-D (in which case the norm for each row is checked).
        epsilon : float
            Parameter for numerical stability, each dimension of the vector is reduced by `epsilon`
            if the norm of the vector is greater than or equal to 1.

        Returns
        -------
        numpy.array
            Array with norms clipped below 1.

        """
        one_d = len(vectors.shape) == 1
        threshold = 1 - epsilon
        if one_d:
            norm = np.linalg.norm(vectors)
            if norm < threshold:
                return vectors
            else:
                return vectors / norm - (np.sign(vectors) * epsilon)
        else:
            norms = np.linalg.norm(vectors, axis=1)
            if (norms < threshold).all():
                return vectors
            else:
                vectors[norms >= threshold] *= (threshold / norms[norms >= threshold])[:, np.newaxis]
                vectors[norms >= threshold] -= np.sign(vectors[norms >= threshold]) * epsilon
                return vectors

    def save(self, *args, **kwargs):
        """Save complete model to disk, inherited from :class:`~gensim.utils.SaveLoad`.

        See also
        --------
        :meth:`~gensim.models.poincare.PoincareModel.load`

        Parameters
        ----------
        *args
            Positional arguments passed to :meth:`~gensim.utils.SaveLoad.save`.
        **kwargs
            Keyword arguments passed to :meth:`~gensim.utils.SaveLoad.save`.

        """
        self._loss_grad = None  # Can't pickle autograd fn to disk
        attrs_to_ignore = ['_node_probabilities', '_node_counts_cumsum']
        kwargs['ignore'] = set(list(kwargs.get('ignore', [])) + attrs_to_ignore)
        super(PoincareModel, self).save(*args, **kwargs)

    @classmethod
    def load(cls, *args, **kwargs):
        """Load model from disk, inherited from :class:`~gensim.utils.SaveLoad`.

        See also
        --------
        :meth:`~gensim.models.poincare.PoincareModel.save`

        Parameters
        ----------
        *args
            Positional arguments passed to :meth:`~gensim.utils.SaveLoad.load`.
        **kwargs
            Keyword arguments passed to :meth:`~gensim.utils.SaveLoad.load`.

        Returns
        -------
        :class:`~gensim.models.poincare.PoincareModel`
            The loaded model.

        """
        model = super(PoincareModel, cls).load(*args, **kwargs)
        model._init_node_probabilities()
        return model

    def _prepare_training_batch(self, relations):
        """Create a training batch and compute gradients and loss for the batch.

        Parameters
        ----------
        relations : list of tuples
            List of tuples of positive examples of the form (node_1_index, node_2_index).

        Returns
        -------
        :class:`~gensim.models.poincare.PoincareBatch`
            Node indices, computed gradients and loss for the batch.

        """
        vectors_u = self.kv.syn0[self.indices_u]
        vectors_v = self.kv.syn0[self.indices_v].reshape((len(relations), 1, self.size))
        vectors_v = vectors_v.swapaxes(0, 1).swapaxes(1, 2)
        batch = PoincareBatch(vectors_u, vectors_v, self.indices_u, \
        self.indices_v, self.new_weights, self.regularization_coeff)
        batch.compute_all()
        return batch

    @staticmethod
    def _handle_duplicates(vector_updates, node_indices):
        """Handle occurrences of multiple updates to the same node in a batch of vector updates.

        Parameters
        ----------
        vector_updates : numpy.array
            Array with each row containing updates to be performed on a certain node.
        node_indices : list of int
            Node indices on which the above updates are to be performed on.

        Notes
        -----
        Mutates the `vector_updates` array.

        Required because vectors[[2, 1, 2]] += np.array([-0.5, 1.0, 0.5]) performs only the last update
        on the row at index 2.

        """
        counts = Counter(node_indices)
        for node_index, count in counts.items():
            if count == 1:
                continue
            # print("node_index", node_index)
            positions = [i for i, index in enumerate(node_indices) if index == node_index]
            # Move all updates to the same node to the last such update, zeroing all the others
            vector_updates[positions[-1]] = vector_updates[positions].sum(axis=0)
            vector_updates[positions[:-1]] = 0

    def _update_vectors_batch(self, batch):
        """Update vectors for nodes in the given batch.

        Parameters
        ----------
        batch : :class:`~gensim.models.poincare.PoincareBatch`
            Batch containing computed gradients and node indices of the batch for which updates are to be done.

        """
        grad_u, grad_v = batch.gradients_u, batch.gradients_v
        indices_u, indices_v = batch.indices_u, batch.indices_v
        batch_size = len(indices_u)

        u_updates = (self.alpha * (batch.alpha ** 2) / 4 * grad_u).T
        node_indexs = list(self.positions_u.keys())
        for node_index in node_indexs:
            self.kv.syn0[node_index,:] -= u_updates[self.positions_u[node_index]].sum(axis=0)

        if not (self.root == None):
            self.kv.syn0[self.kv.index2word.index(self.root)] = 0

        self.kv.syn0 = self._clip_vectors(self.kv.syn0, self.epsilon)

        v_updates = self.alpha * (batch.beta ** 2)[:, np.newaxis] / 4 * grad_v
        v_updates = v_updates.swapaxes(1, 2).swapaxes(0, 1)
        v_updates = v_updates.reshape((batch_size, self.size))
        node_indexs = list(self.positions_v.keys())
        for node_index in node_indexs:
            self.kv.syn0[node_index,:] -= v_updates[self.positions_v[node_index]].sum(axis=0)

        if not (self.root == None):
            self.kv.syn0[self.kv.index2word.index(self.root)] = 0

        self.kv.syn0 = self._clip_vectors(self.kv.syn0, self.epsilon)

    def train(self, epochs, print_every=1000):
        """Train Poincare embeddings using loaded data and model parameters.

        Parameters
        ----------
        epochs : int
            Number of iterations (epochs) over the corpus.

        print_every : int, optional
            Prints progress and average loss after every `print_every` batches.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.models.poincare import PoincareModel
            >>> relations = [('kangaroo', 'marsupial'), ('kangaroo', 'mammal'), ('gib', 'cat')]
            >>> model = PoincareModel(relations)
            >>> model.train(epochs=50)

        """

        if self.workers > 1:
            raise NotImplementedError("Multi-threaded version not implemented yet")
        # Some divide-by-zero results are handled explicitly
        old_settings = np.seterr(divide='ignore', invalid='ignore')

        logger.info(
            "training model of size %d with %d workers on %d relations for %d epochs and %d burn-in epochs, "
            "using lr=%.5f burn-in lr=%.5f",
            self.size, self.workers, len(self.all_relations), epochs, self.burn_in,
            self.alpha, self.burn_in_alpha)

        if self.burn_in > 0 and not self._burn_in_done:
            logger.info("starting burn-in (%d epochs)----------------------------------------", self.burn_in)
            self.alpha = self.burn_in_alpha
            self._train_batchwise(
                epochs=self.burn_in, print_every=print_every)
            self._burn_in_done = True
            logger.info("burn-in finished")

        self.alpha = self.train_alpha
        logger.info("starting training (%d epochs)----------------------------------------", epochs)
        self._train_batchwise(
            epochs=epochs, print_every=print_every)
        logger.info("training finished")

        np.seterr(**old_settings)

    def step_decay(self, alpha, epoch):
        initial_lrate = alpha
        drop = 1.1
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate

    def _train_batchwise(self, epochs, print_every=1000):
        """Train Poincare embeddings using specified parameters.

        Parameters
        ----------
        epochs : int
            Number of iterations (epochs) over the corpus.
        print_every : int, optional
            Prints progress and average loss after every `print_every` batches.

        """
        start_time = time.time()
        if self.workers > 1:
            raise NotImplementedError("Multi-threaded version not implemented yet")
        indices = list(range(len(self.all_relations)))
        self._np_random.shuffle(indices)
        relations = [self.all_relations[idx] for idx in indices]
        # print("relations:", relations)
        b = list(self.indices_set)

        print("OK1")

        indices_u, indices_v, all_weights = [], [], []
        for relation in relations:
            weights = []
            u, v = relation
            indices_u.append(u)
            indices_v.append(v)
            weights.append(self.weights[(u,v)])
            all_weights.append(weights)
        self.indices_u = indices_u
        self.indices_v = indices_v
        self.new_weights = np.asarray(all_weights).T

        print("OK2")

        counts = list(set(indices_u))
        self.positions_u = dict()
        for node_index in counts:
            self.positions_u[node_index] = [i for i, index in enumerate(indices_u) \
            if index == node_index]

        print("OK3")

        counts = list(set(indices_v))
        self.positions_v = dict()
        for node_index in counts:
            self.positions_v[node_index] = [i for i, index in enumerate(indices_v) \
            if index == node_index]

        print("OK4")

        # data = np.load("data_file.npz", allow_pickle=True)
        # self.indices_u = data["indices_u"].tolist()
        # self.indices_v = data["indices_v"].tolist()
        # self.new_weights = data["new_weights"]
        # self.positions_u = data["positions_u"].item()
        # self.positions_v = data["positions_v"].item()

        # print(type(self.positions_u))
        #
        # print("ready!")
        for epoch in range(1, epochs + 1):
            batch = self._prepare_training_batch(relations)
            if epoch % int(print_every) == 0:
                print(batch.loss)
            self._update_vectors_batch(batch)

class PoincareBatch(object):
    """Compute Poincare distances, gradients and loss for a training batch.

    Store intermediate state to avoid recomputing multiple times.

    """
    def __init__(self, vectors_u, vectors_v, indices_u, indices_v, weights, regularization_coeff=1.0):
        """
        Initialize instance with sets of vectors for which distances are to be computed.

        Parameters
        ----------
        vectors_u : numpy.array
            Vectors of all nodes `u` in the batch. Expected shape (batch_size, dim).
        vectors_v : numpy.array
            Vectors of all positively related nodes `v` and negatively sampled nodes `v'`,
            for each node `u` in the batch. Expected shape (1 + neg_size, dim, batch_size).
        indices_u : list of int
            List of node indices for each of the vectors in `vectors_u`.
        indices_v : list of lists of int
            Nested list of lists, each of which is a  list of node indices
            for each of the vectors in `vectors_v` for a specific node `u`.
        regularization_coeff : float, optional
            Coefficient to use for l2-regularization

        """
        self.vectors_u = vectors_u.T[np.newaxis, :, :]  # (1, dim, batch_size)
        self.vectors_v = vectors_v  # (1 + neg_size, dim, batch_size)
        self.indices_u = indices_u
        self.weights = weights
        self.indices_v = indices_v
        self.regularization_coeff = regularization_coeff

        self.poincare_dists = None
        self.euclidean_dists = None

        self.norms_u = None
        self.norms_v = None
        self.alpha = None
        self.beta = None
        self.gamma = None

        self.gradients_u = None
        self.distance_gradients_u = None
        self.gradients_v = None
        self.distance_gradients_v = None

        self.loss = None

        self._distances_computed = False
        self._gradients_computed = False
        self._distance_gradients_computed = False
        self._loss_computed = False

    def compute_all(self):
        """Convenience method to perform all computations."""
        self.compute_distances()
        self.compute_distance_gradients()
        self.compute_gradients()
        self.compute_loss()

    def compute_distances(self):
        """Compute and store norms, euclidean distances and poincare distances between input vectors."""
        if self._distances_computed:
            return
        euclidean_dists = np.linalg.norm(self.vectors_u - self.vectors_v, axis=1)  # (1 + neg_size, batch_size)
        norms_u = np.linalg.norm(self.vectors_u, axis=1)  # (1, batch_size)
        norms_v = np.linalg.norm(self.vectors_v, axis=1)  # (1 + neg_size, batch_size)
        alpha = 1 - norms_u ** 2  # (1, batch_size)
        beta = 1 - norms_v ** 2  # (1 + neg_size, batch_size)
        gamma = 1 + 2 * (
                (euclidean_dists ** 2) / (alpha * beta)
            )  # (1 + neg_size, batch_size)
        poincare_dists = np.arccosh(gamma)  # (1 + neg_size, batch_size)
        new_square_distances = (poincare_dists / self.weights - 1) ** 2
        new_Z = new_square_distances.sum(axis=0)
        self.euclidean_dists = euclidean_dists
        self.poincare_dists = poincare_dists
        self.new_square_distances = new_square_distances
        self.new_Z = new_Z
        self.gamma = gamma
        self.norms_u = norms_u
        self.norms_v = norms_v
        self.alpha = alpha
        self.beta = beta

        self._distances_computed = True

    def compute_gradients(self):
        """Compute and store gradients of loss function for all input vectors."""
        if self._gradients_computed:
            return
        self.compute_distances()
        self.compute_distance_gradients()

        diff = (self.poincare_dists / self.weights - 1)
        gradients_v = 1 / self.weights[:, np.newaxis, :] * \
        2 * diff[:, np.newaxis, :] * self.distance_gradients_v
        gradients_v[0] += self.regularization_coeff * 2 * self.vectors_v[0]

        # (1 + neg_size, dim, batch_size)
        gradients_u = 1 / self.weights[:, np.newaxis, :] * \
        2 * diff[:, np.newaxis, :] * self.distance_gradients_u
        gradients_u = gradients_u.sum(axis=0)  # (dim, batch_size)
        if np.isnan(gradients_u).any() or np.isnan(gradients_v).any():
            print("gradients_u:", gradients_u)
            print("gradients_v:", gradients_v)
            sys.exit()
        self.gradients_u = gradients_u
        self.gradients_v = gradients_v

        self._gradients_computed = True

    def compute_distance_gradients(self):
        """Compute and store partial derivatives of poincare distance d(u, v) w.r.t all u and all v."""
        if self._distance_gradients_computed:
            return
        self.compute_distances()

        euclidean_dists_squared = self.euclidean_dists ** 2  # (1, batch_size)
        # (1, 1, batch_size)
        c_ = (4 / (self.alpha * self.beta * np.sqrt(self.gamma ** 2 - 1)))[:, np.newaxis, :]
        # (1, 1, batch_size)
        u_coeffs = ((euclidean_dists_squared + self.alpha) / self.alpha)[:, np.newaxis, :]
        distance_gradients_u = u_coeffs * self.vectors_u - self.vectors_v  # (1, dim, batch_size)
        distance_gradients_u *= c_  # (1, dim, batch_size)

        nan_gradients = self.gamma == 1  # (1, batch_size)
        if nan_gradients.any():
            distance_gradients_u.swapaxes(1, 2)[nan_gradients] = 0
        self.distance_gradients_u = distance_gradients_u

        # (1, 1, batch_size)
        v_coeffs = ((euclidean_dists_squared + self.beta) / self.beta)[:, np.newaxis, :]
        distance_gradients_v = v_coeffs * self.vectors_v - self.vectors_u  # (1, dim, batch_size)
        distance_gradients_v *= c_  # (1, dim, batch_size)

        if nan_gradients.any():
            distance_gradients_v.swapaxes(1, 2)[nan_gradients] = 0
        self.distance_gradients_v = distance_gradients_v

        self._distance_gradients_computed = True

    def compute_loss(self):
        """Compute and store loss value for the given batch of examples."""
        if self._loss_computed:
            return
        self.compute_distances()
        self.loss = self.new_Z.sum()  # scalar
        self._loss_computed = True


class PoincareKeyedVectors(BaseKeyedVectors):
    """Vectors and vocab for the :class:`~gensim.models.poincare.PoincareModel` training class.

    Used to perform operations on the vectors such as vector lookup, distance calculations etc.

    """
    def __init__(self, vector_size):
        super(PoincareKeyedVectors, self).__init__(vector_size)
        self.max_distance = 0
        self.index2word = []
        self.vocab = {}

    @property
    def vectors(self):
        return self.syn0

    @vectors.setter
    def vectors(self, value):
        self.syn0 = value

    @property
    def index2entity(self):
        return self.index2word

    @index2entity.setter
    def index2entity(self, value):
        self.index2word = value

    def word_vec(self, word):
        """Get the word's representations in vector space, as a 1D numpy array.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>>
            >>> # Read the sample relations file and train the model
            >>> relations = PoincareRelations(file_path=datapath('poincare_hypernyms_large.tsv'))
            >>> model = PoincareModel(train_data=relations)
            >>> model.train(epochs=50)
            >>>
            >>> # Query the trained model.
            >>> wv = model.kv.word_vec('kangaroo.n.01')

        """
        return super(PoincareKeyedVectors, self).get_vector(word)

    def words_closer_than(self, w1, w2):
        """Get all words that are closer to `w1` than `w2` is to `w1`.

        Parameters
        ----------
        w1 : str
            Input word.
        w2 : str
            Input word.

        Returns
        -------
        list (str)
            List of words that are closer to `w1` than `w2` is to `w1`.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>>
            >>> # Read the sample relations file and train the model
            >>> relations = PoincareRelations(file_path=datapath('poincare_hypernyms_large.tsv'))
            >>> model = PoincareModel(train_data=relations)
            >>> model.train(epochs=50)
            >>>
            >>> # Which term is closer to 'kangaroo' than 'metatherian' is to 'kangaroo'?
            >>> model.kv.words_closer_than('kangaroo.n.01', 'metatherian.n.01')
            [u'marsupial.n.01', u'phalanger.n.01']

        """
        return super(PoincareKeyedVectors, self).closer_than(w1, w2)

    def save_word2vec_format(self, fname, fvocab=None, binary=False, total_vec=None):
        """Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility, using :func:`~gensim.models.utils_any2vec._save_word2vec_format`.

        Parameters
        ----------
        fname : str
            Path to file that will be used for storing.
        fvocab : str, optional
            File path used to save the vocabulary.
        binary : bool, optional
            If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.
        total_vec : int, optional
            Explicitly specify total number of vectors
            (in case word vectors are appended with document vectors afterwards).

        """
        _save_word2vec_format(fname, self.vocab, self.syn0, fvocab=fvocab, binary=binary, total_vec=total_vec)

    @classmethod
    def load_word2vec_format(cls, fname, fvocab=None, binary=False, encoding='utf8', unicode_errors='strict',
                             limit=None, datatype=REAL):
        """Load the input-hidden weight matrix from the original C word2vec-tool format.
        Use :func:`~gensim.models.utils_any2vec._load_word2vec_format`.

        Note that the information stored in the file is incomplete (the binary tree is missing),
        so while you can query for word similarity etc., you cannot continue training
        with a model loaded this way.

        Parameters
        ----------
        fname : str
            The file path to the saved word2vec-format file.
        fvocab : str, optional
            File path to the vocabulary.Word counts are read from `fvocab` filename, if set
            (this is the file generated by `-save-vocab` flag of the original C tool).
        binary : bool, optional
            If True, indicates whether the data is in binary word2vec format.
        encoding : str, optional
            If you trained the C model using non-utf8 encoding for words, specify that encoding in `encoding`.
        unicode_errors : str, optional
            default 'strict', is a string suitable to be passed as the `errors`
            argument to the unicode() (Python 2.x) or str() (Python 3.x) function. If your source
            file may include word tokens truncated in the middle of a multibyte unicode character
            (as is common from the original word2vec.c tool), 'ignore' or 'replace' may help.
        limit : int, optional
            Sets a maximum number of word-vectors to read from the file. The default,
            None, means read all.
        datatype : type, optional
            (Experimental) Can coerce dimensions to a non-default float type (such as `np.float16`) to save memory.
            Such types may result in much slower bulk operations or incompatibility with optimized routines.)

        Returns
        -------
        :class:`~gensim.models.poincare.PoincareModel`
            Loaded Poincare model.

        """
        return _load_word2vec_format(
            cls, fname, fvocab=fvocab, binary=binary, encoding=encoding, unicode_errors=unicode_errors,
            limit=limit, datatype=datatype)

    @staticmethod
    def vector_distance(vector_1, vector_2):
        """Compute poincare distance between two input vectors. Convenience method over `vector_distance_batch`.

        Parameters
        ----------
        vector_1 : numpy.array
            Input vector.
        vector_2 : numpy.array
            Input vector.

        Returns
        -------
        numpy.float
            Poincare distance between `vector_1` and `vector_2`.

        """
        return PoincareKeyedVectors.vector_distance_batch(vector_1, vector_2[np.newaxis, :])[0]

    @staticmethod
    def vector_distance_batch(vector_1, vectors_all):
        """Compute poincare distances between one vector and a set of other vectors.

        Parameters
        ----------
        vector_1 : numpy.array
            vector from which Poincare distances are to be computed, expected shape (dim,).
        vectors_all : numpy.array
            for each row in vectors_all, distance from vector_1 is computed, expected shape (num_vectors, dim).

        Returns
        -------
        numpy.array
            Poincare distance between `vector_1` and each row in `vectors_all`, shape (num_vectors,).

        """
        euclidean_dists = np.linalg.norm(vector_1 - vectors_all, axis=1)
        norm = np.linalg.norm(vector_1)
        all_norms = np.linalg.norm(vectors_all, axis=1)
        return np.arccosh(
            1 + 2 * (
                (euclidean_dists ** 2) / ((1 - norm ** 2) * (1 - all_norms ** 2))
            )
        )

    def closest_child(self, node):
        """Get the node closest to `node` that is lower in the hierarchy than `node`.

        Parameters
        ----------
        node : {str, int}
            Key for node for which closest child is to be found.

        Returns
        -------
        {str, None}
            Node closest to `node` that is lower in the hierarchy than `node`.
            If there are no nodes lower in the hierarchy, None is returned.

        """
        all_distances = self.distances(node)
        all_norms = np.linalg.norm(self.syn0, axis=1)
        node_norm = all_norms[self.vocab[node].index]
        mask = node_norm >= all_norms
        if mask.all():  # No nodes lower in the hierarchy
            return None
        all_distances = np.ma.array(all_distances, mask=mask)
        closest_child_index = np.ma.argmin(all_distances)
        return self.index2word[closest_child_index]

    def closest_parent(self, node):
        """Get the node closest to `node` that is higher in the hierarchy than `node`.

        Parameters
        ----------
        node : {str, int}
            Key for node for which closest parent is to be found.

        Returns
        -------
        {str, None}
            Node closest to `node` that is higher in the hierarchy than `node`.
            If there are no nodes higher in the hierarchy, None is returned.

        """
        all_distances = self.distances(node)
        all_norms = np.linalg.norm(self.syn0, axis=1)
        node_norm = all_norms[self.vocab[node].index]
        mask = node_norm <= all_norms
        if mask.all():  # No nodes higher in the hierarchy
            return None
        all_distances = np.ma.array(all_distances, mask=mask)
        closest_child_index = np.ma.argmin(all_distances)
        return self.index2word[closest_child_index]

    def descendants(self, node, max_depth=5):
        """Get the list of recursively closest children from the given node, up to a max depth of `max_depth`.

        Parameters
        ----------
        node : {str, int}
            Key for node for which descendants are to be found.
        max_depth : int
            Maximum number of descendants to return.

        Returns
        -------
        list of str
            Descendant nodes from the node `node`.

        """
        depth = 0
        descendants = []
        current_node = node
        while depth < max_depth:
            descendants.append(self.closest_child(current_node))
            current_node = descendants[-1]
            depth += 1
        return descendants

    def ancestors(self, node):
        """Get the list of recursively closest parents from the given node.

        Parameters
        ----------
        node : {str, int}
            Key for node for which ancestors are to be found.

        Returns
        -------
        list of str
            Ancestor nodes of the node `node`.

        """
        ancestors = []
        current_node = node
        ancestor = self.closest_parent(current_node)
        while ancestor is not None:
            ancestors.append(ancestor)
            ancestor = self.closest_parent(ancestors[-1])
        return ancestors

    def distance(self, w1, w2):
        """Calculate Poincare distance between vectors for nodes `w1` and `w2`.

        Parameters
        ----------
        w1 : {str, int}
            Key for first node.
        w2 : {str, int}
            Key for second node.

        Returns
        -------
        float
            Poincare distance between the vectors for nodes `w1` and `w2`.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>>
            >>> # Read the sample relations file and train the model
            >>> relations = PoincareRelations(file_path=datapath('poincare_hypernyms_large.tsv'))
            >>> model = PoincareModel(train_data=relations)
            >>> model.train(epochs=50)
            >>>
            >>> # What is the distance between the words 'mammal' and 'carnivore'?
            >>> model.kv.distance('mammal.n.01', 'carnivore.n.01')
            2.9742298803339304

        Raises
        ------
        KeyError
            If either of `w1` and `w2` is absent from vocab.

        """
        vector_1 = self.word_vec(w1)
        vector_2 = self.word_vec(w2)
        return self.vector_distance(vector_1, vector_2)

    def similarity(self, w1, w2):
        """Compute similarity based on Poincare distance between vectors for nodes `w1` and `w2`.

        Parameters
        ----------
        w1 : {str, int}
            Key for first node.
        w2 : {str, int}
            Key for second node.

        Returns
        -------
        float
            Similarity between the between the vectors for nodes `w1` and `w2` (between 0 and 1).

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>>
            >>> # Read the sample relations file and train the model
            >>> relations = PoincareRelations(file_path=datapath('poincare_hypernyms_large.tsv'))
            >>> model = PoincareModel(train_data=relations)
            >>> model.train(epochs=50)
            >>>
            >>> # What is the similarity between the words 'mammal' and 'carnivore'?
            >>> model.kv.similarity('mammal.n.01', 'carnivore.n.01')
            0.25162107631176484

        Raises
        ------
        KeyError
            If either of `w1` and `w2` is absent from vocab.

        """
        return 1 / (1 + self.distance(w1, w2))

    def most_similar(self, node_or_vector, topn=10, restrict_vocab=None):
        """Find the top-N most similar nodes to the given node or vector, sorted in increasing order of distance.

        Parameters
        ----------
        node_or_vector : {str, int, numpy.array}
            node key or vector for which similar nodes are to be found.
        topn : int or None, optional
            Number of top-N similar nodes to return, when `topn` is int. When `topn` is None,
            then distance for all nodes are returned.
        restrict_vocab : int or None, optional
            Optional integer which limits the range of vectors which are searched for most-similar values.
            For example, restrict_vocab=10000 would only check the first 10000 node vectors in the vocabulary order.
            This may be meaningful if vocabulary is sorted by descending frequency.

        Returns
        --------
        list of (str, float) or numpy.array
            When `topn` is int, a sequence of (node, distance) is returned in increasing order of distance.
            When `topn` is None, then similarities for all words are returned as a one-dimensional numpy array with the
            size of the vocabulary.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>>
            >>> # Read the sample relations file and train the model
            >>> relations = PoincareRelations(file_path=datapath('poincare_hypernyms_large.tsv'))
            >>> model = PoincareModel(train_data=relations)
            >>> model.train(epochs=50)
            >>>
            >>> # Which words are most similar to 'kangaroo'?
            >>> model.kv.most_similar('kangaroo.n.01', topn=2)
            [(u'kangaroo.n.01', 0.0), (u'marsupial.n.01', 0.26524229460827725)]

        """
        if isinstance(topn, Integral) and topn < 1:
            return []

        if not restrict_vocab:
            all_distances = self.distances(node_or_vector)
        else:
            nodes_to_use = self.index2word[:restrict_vocab]
            all_distances = self.distances(node_or_vector, nodes_to_use)

        if isinstance(node_or_vector, string_types + (int,)):
            node_index = self.vocab[node_or_vector].index
        else:
            node_index = None
        if not topn:
            closest_indices = matutils.argsort(all_distances)
        else:
            closest_indices = matutils.argsort(all_distances, topn=1 + topn)
        result = [
            (self.index2word[index], float(all_distances[index]))
            for index in closest_indices if (not node_index or index != node_index)  # ignore the input node
        ]
        if topn:
            result = result[:topn]
        return result

    def distances(self, node_or_vector, other_nodes=()):
        """Compute Poincare distances from given `node_or_vector` to all nodes in `other_nodes`.
        If `other_nodes` is empty, return distance between `node_or_vector` and all nodes in vocab.

        Parameters
        ----------
        node_or_vector : {str, int, numpy.array}
            Node key or vector from which distances are to be computed.
        other_nodes : {iterable of str, iterable of int, None}, optional
            For each node in `other_nodes` distance from `node_or_vector` is computed.
            If None or empty, distance of `node_or_vector` from all nodes in vocab is computed (including itself).

        Returns
        -------
        numpy.array
            Array containing distances to all nodes in `other_nodes` from input `node_or_vector`,
            in the same order as `other_nodes`.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>>
            >>> # Read the sample relations file and train the model
            >>> relations = PoincareRelations(file_path=datapath('poincare_hypernyms_large.tsv'))
            >>> model = PoincareModel(train_data=relations)
            >>> model.train(epochs=50)
            >>>
            >>> # Check the distances between a word and a list of other words.
            >>> model.kv.distances('mammal.n.01', ['carnivore.n.01', 'dog.n.01'])
            array([2.97422988, 2.83007402])

            >>> # Check the distances between a word and every other word in the vocab.
            >>> all_distances = model.kv.distances('mammal.n.01')

        Raises
        ------
        KeyError
            If either `node_or_vector` or any node in `other_nodes` is absent from vocab.

        """
        if isinstance(node_or_vector, string_types):
            input_vector = self.word_vec(node_or_vector)
        else:
            input_vector = node_or_vector
        if not other_nodes:
            other_vectors = self.syn0
        else:
            other_indices = [self.vocab[node].index for node in other_nodes]
            other_vectors = self.syn0[other_indices]
        return self.vector_distance_batch(input_vector, other_vectors)

    def norm(self, node_or_vector):
        """Compute absolute position in hierarchy of input node or vector.
        Values range between 0 and 1. A lower value indicates the input node or vector is higher in the hierarchy.

        Parameters
        ----------
        node_or_vector : {str, int, numpy.array}
            Input node key or vector for which position in hierarchy is to be returned.

        Returns
        -------
        float
            Absolute position in the hierarchy of the input vector or node.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>>
            >>> # Read the sample relations file and train the model
            >>> relations = PoincareRelations(file_path=datapath('poincare_hypernyms_large.tsv'))
            >>> model = PoincareModel(train_data=relations)
            >>> model.train(epochs=50)
            >>>
            >>> # Get the norm of the embedding of the word `mammal`.
            >>> model.kv.norm('mammal.n.01')
            0.6423008703542398

        Notes
        -----
        The position in hierarchy is based on the norm of the vector for the node.

        """
        if isinstance(node_or_vector, string_types):
            input_vector = self.word_vec(node_or_vector)
        else:
            input_vector = node_or_vector
        return np.linalg.norm(input_vector)

    def difference_in_hierarchy(self, node_or_vector_1, node_or_vector_2):
        """Compute relative position in hierarchy of `node_or_vector_1` relative to `node_or_vector_2`.
        A positive value indicates `node_or_vector_1` is higher in the hierarchy than `node_or_vector_2`.

        Parameters
        ----------
        node_or_vector_1 : {str, int, numpy.array}
            Input node key or vector.
        node_or_vector_2 : {str, int, numpy.array}
            Input node key or vector.

        Returns
        -------
        float
            Relative position in hierarchy of `node_or_vector_1` relative to `node_or_vector_2`.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>>
            >>> # Read the sample relations file and train the model
            >>> relations = PoincareRelations(file_path=datapath('poincare_hypernyms_large.tsv'))
            >>> model = PoincareModel(train_data=relations)
            >>> model.train(epochs=50)
            >>>
            >>> model.kv.difference_in_hierarchy('mammal.n.01', 'dog.n.01')
            0.05382517902410999

            >>> model.kv.difference_in_hierarchy('dog.n.01', 'mammal.n.01')
            -0.05382517902410999

        Notes
        -----
        The returned value can be positive or negative, depending on whether `node_or_vector_1` is higher
        or lower in the hierarchy than `node_or_vector_2`.

        """
        return self.norm(node_or_vector_2) - self.norm(node_or_vector_1)


class PoincareRelations(object):
    """Stream relations for `PoincareModel` from a tsv-like file."""

    def __init__(self, file_path, encoding='utf8', delimiter=','):
        """Initialize instance from file containing a pair of nodes (a relation) per line.

        Parameters
        ----------
        file_path : str
            Path to file containing a pair of nodes (a relation) per line, separated by `delimiter`.
            Since the relations are asymmetric, the order of `u` and `v` nodes in each pair matters.
            To express a "u is v" relation, the lines should take the form `u delimeter v`.
            e.g: `kangaroo	mammal` is a tab-delimited line expressing a "`kangaroo is a mammal`" relation.

            For a full input file example, see `gensim/test/test_data/poincare_hypernyms.tsv
            <https://github.com/RaRe-Technologies/gensim/blob/master/gensim/test/test_data/poincare_hypernyms.tsv>`_.
        encoding : str, optional
            Character encoding of the input file.
        delimiter : str, optional
            Delimiter character for each relation.

        """

        self.file_path = file_path
        self.encoding = encoding
        self.delimiter = delimiter

    def __iter__(self):
        """Stream relations from self.file_path decoded into unicode strings.

        Yields
        -------
        (unicode, unicode)
            Relation from input file.

        """
        with utils.open(self.file_path, 'rb') as file_obj:
            if sys.version_info[0] < 3:
                lines = file_obj
            else:
                lines = (l.decode(self.encoding) for l in file_obj)
            # csv.reader requires bytestring input in python2, unicode input in python3
            reader = csv.reader(lines, delimiter=self.delimiter)
            for row in reader:
                if sys.version_info[0] < 3:
                    row = [value.decode(self.encoding) for value in row]
                yield tuple(row)

# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BLEU metric util used during eval for MT."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

# Dependency imports

import numpy as np
# pylint: disable=redefined-builtin
from six.moves import xrange
from six.moves import zip
from collections import defaultdict
# pylint: enable=redefined-builtin

import tensorflow as tf

PAD_ID = 0
EOS_ID = 1
SPECIAL_TOKENS=[PAD_ID, EOS_ID]

def _get_ngrams(segment, max_order):
  """Extracts all n-grams upto a given maximum order from an input segment.

  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.

  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_counts = collections.Counter()
  for order in xrange(1, max_order + 1):
    for i in xrange(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i + order])
      ngram_counts[ngram] += 1
  return ngram_counts


def compute_bleu(reference_corpus,
                 translation_corpus,
                 max_order=4,
                 use_bp=True):
  """Computes BLEU score of translated segments against one or more references.

  Args:
    reference_corpus: list of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    use_bp: boolean, whether to apply brevity penalty.

  Returns:
    BLEU score.
  """
  reference_length = 0
  translation_length = 0
  bp = 1.0
  geo_mean = 0

  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  precisions = []

  for (references, translations) in zip(reference_corpus, translation_corpus):
    reference_length += len(references)    # length of each sentence
    translation_length += len(translations)
    ref_ngram_counts = _get_ngrams(references, max_order)
    translation_ngram_counts = _get_ngrams(translations, max_order)

    overlap = dict((ngram,
                    min(count, translation_ngram_counts[ngram]))
                   for ngram, count in ref_ngram_counts.items())

    for ngram in overlap:
      matches_by_order[len(ngram) - 1] += overlap[ngram]
    for ngram in translation_ngram_counts:
      possible_matches_by_order[len(ngram)-1] += translation_ngram_counts[ngram]
  precisions = [0] * max_order
  for i in xrange(0, max_order):
    if possible_matches_by_order[i] > 0:
      precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
    else:
      precisions[i] = 0.0

  if max(precisions) > 0:
    p_log_sum = sum(math.log(p) for p in precisions if p)
    geo_mean = math.exp(p_log_sum/max_order)

  if use_bp:
    ratio = translation_length / reference_length
    bp = math.exp(1 - 1. / ratio) if ratio < 1.0 else 1.0
  bleu = geo_mean * bp
  return np.float32(bleu)


def _save_until_pad(ref):
    """Search for the first <pad> position in reference sentence."""
    ref = ref.flatten()
    try:
        index = list(ref).index(PAD_ID)
        return ref[0:index]
    except ValueError:
        # No PAD_ID: return the array as-is
        return ref


def _save_until_eos(hyp):
    """Strips everything after the first <EOS> token, which is normally 1."""
    # sample will not contain any <pad>, only <eos>
    hyp = hyp.flatten()
    try:
        index = list(hyp).index(EOS_ID)
        return hyp[0:index]
    except ValueError:
        # No EOS_ID: return the array as-is.
        return hyp


def _bleu(y, y_hat, n=4):
    """ BLEU score between the reference sequence y
    and y_hat for each partial sequence ranging
    from the first input token to the last

    Parameters
    ----------
    y : vector
        The reference matrix with dimensions of number
        of words (rows) by batch size (columns)
    y_hat : vector
        The predicted matrix with same dimensions
    n : integer
        highest n-gram order in the Bleu sense
        (e.g Bleu-4)

    Returns
    -------
    results : vector (len y_hat)
        Bleu scores for each partial sequence
        y_hat_1..T from T = 1 to len(y_hat)
    """
    bleu_scores = np.zeros((len(y_hat), n))

    # count reference ngrams
    ref_counts = defaultdict(int)
    for k in xrange(1, n+1):
        for i in xrange(len(y) - k + 1):
            ref_counts[tuple(y[i:i + k])] += 1

    # for each partial sequence, 1) compute addition to # of correct
    # 2) apply brevity penalty
    # ngrams, magic stability numbers from pycocoeval
    ref_len = len(y)
    pred_counts = defaultdict(int)
    correct = np.zeros(4)
    for i in xrange(1, len(y_hat) + 1):
        for k in xrange(i, max(-1, i - n), -1):
            # print i, k
            ngram = tuple(y_hat[k-1:i])
            # UNK token hack. Must work for both indices
            # and words. Very ugly, I know.
            if 0 in ngram or 'UNK' in ngram:
                continue
            pred_counts[ngram] += 1
            if pred_counts[ngram] <= ref_counts.get(ngram, 0):
                correct[len(ngram)-1] += 1

        # compute partial bleu score
        bleu = 1.
        for j in xrange(n):
            possible = max(0, i - j)
            bleu *= float(correct[j] + 1.) / (possible + 1.)
            bleu_scores[i - 1, j] = bleu ** (1./(j+1))

        # brevity penalty
        if i < ref_len:
            ratio = (i + 1e-15)/(ref_len + 1e-9)
            bleu_scores[i - 1, :] *= math.exp(1 - 1/ratio)

    return bleu_scores.astype('float32'), correct, pred_counts, ref_counts


def compute_sentence_bleu(reference_batch, translation_batch, max_order=4):
    delta_results = np.zeros_like(translation_batch).astype('float32')  # [batch, times]
    batch_size = reference_batch.shape[0]
    for index in range(batch_size):
        reference = reference_batch[index, :]
        translation = translation_batch[index, :]
        # reference = _save_until_pad(reference)  # remove <pad> for reference
        reference = [token for token in reference if token not in SPECIAL_TOKENS]
        translation = [token for token in translation if token not in SPECIAL_TOKENS]

        bleu_scores, _, _, _ = _bleu(reference, translation, max_order)

        reward = bleu_scores[:, max_order - 1].copy()
        # delta rewards
        reward[1:] = reward[1:] - reward[:-1]
        pos = -1
        for i in range(len(translation)):
            if translation[i] not in SPECIAL_TOKENS:
                pos = pos + 1
                delta_results[index, i] = reward[pos]
            else:
                delta_results[index, i] = 0.
        # print(results[index])  # debug
    delta_results = delta_results[::-1].cumsum(axis=1)[::-1]
    return delta_results   # results are delta rewards


def compute_sentence_total_bleu(reference_batch, translation_batch, max_order=4):
    total_results = np.zeros_like(translation_batch).astype('float32')
    batch_size = reference_batch.shape[0]
    for index in range(batch_size):
        reference = reference_batch[index, :]
        translation = translation_batch[index, :]
        # reference = _save_until_pad(reference)  # remove <pad> for reference
        reference = [token for token in reference if token not in SPECIAL_TOKENS]
        translation = [token for token in translation if token not in SPECIAL_TOKENS]

        bleu_scores, _, _, _ = _bleu(reference, translation, max_order)

        reward = bleu_scores[:, max_order - 1].copy()
        # total rewards
        for i in range(0, len(translation)):
            total_results[index, i] = reward[-1]
    # total_results = total_results[::-1].cumsum(axis=1)[::-1]
    return total_results  # results are total


def bleu_score(predictions, labels, **unused_kwargs):
  """BLEU score computation between labels and predictions.

  An approximate BLEU scoring method since we do not glue word pieces or
  decode the ids and tokenize the output. By default, we use ngram order of 4
  and use brevity penalty. Also, this does not have beam search.

  Args:
    predictions: tensor, model predicitons
    labels: tensor, gold output.

  Returns:
    bleu: int, approx bleu score
  """
  outputs = tf.to_int32(tf.argmax(predictions, axis=-1))   # same as greedy infer...
  # Convert the outputs and labels to a [batch_size, input_length] tensor.
  outputs = tf.squeeze(outputs, axis=[-1, -2])
  labels = tf.squeeze(labels, axis=[-1, -2])

  bleu = tf.py_func(compute_bleu, (labels, outputs), tf.float32)
  return bleu, tf.constant(1.0)


def bleu_score_train(predictions, labels, delta_reward=True, **unused_kwargs):
  """BLEU score computation between labels and predictions.

  An approximate BLEU scoring method since we do not glue word pieces or
  decode the ids and tokenize the output. By default, we use ngram order of 4
  and use brevity penalty. Also, this does not have beam search.

  Args:
    predictions: tensor, model predicitons
    labels: tensor, gold output.
    delta_reward: boolean, true to use delta reward, flase to use total reward setting

  Returns:
    bleu: tensor, same shape as predictions, approx bleu score
  """
  # outputs = tf.to_int32(tf.argmax(predictions, axis=-1))   # same as greedy infer...
  # Convert the outputs and labels to a [batch_size, input_length] tensor.
  outputs = tf.squeeze(predictions, axis=[-1, -2])   # predictions are data_ids
  labels = tf.squeeze(labels, axis=[-1, -2])

  if delta_reward:
      bleu = tf.py_func(compute_sentence_bleu, (labels, outputs), tf.float32)  # bleu is delta reward matrix
  else:
      bleu = tf.py_func(compute_sentence_total_bleu, (labels, outputs), tf.float32)  # bleu is delta reward matrix
  return bleu  # , tf.constant(1.0)


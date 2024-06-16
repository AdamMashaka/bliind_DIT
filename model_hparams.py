
"""Hyperparameters for the object detection model in TF.learn.

This file consolidates and documents the hyperparameters used by the model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=g-import-not-at-top
try:
  from tensorflow.contrib import training as contrib_training
except ImportError:
  # TF 2.0 doesn't ship with contrib.
  pass
# pylint: enable=g-import-not-at-top


def create_hparams(hparams_overrides=None):
  """Returns hyperparameters, including any flag value overrides.

  Args:
    hparams_overrides: Optional hparams overrides, represented as a
      string containing comma-separated hparam_name=value pairs.

  Returns:
    The hyperparameters as a tf.HParams object.
  """
  hparams = contrib_training.HParams(
      # Whether a fine tuning checkpoint (provided in the pipeline config)
      # should be loaded for training.
      load_pretrained=True)
  # Override any of the preceding hyperparameter values.
  if hparams_overrides:
    hparams = hparams.parse(hparams_overrides)
  return hparams

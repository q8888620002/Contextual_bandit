
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from bandits.core.bandit_algorithm import BanditAlgorithm

class ClinicalGuideLine(BanditAlgorithm):
  """Defines a baseline; returns one action uniformly at random."""

  def __init__(self, name, hparams):
    """Creates a UniformSampling object.
    Args:
      name: Name of the algorithm.
      hparams: Hyper-parameters, including the number of arms (num_actions).
    """

    self.name = name
    self.hparams = hparams

  def action(self, context):
    """Selects an action uniformly at random."""
    if not self.hparams.guideline_only:
      indices = np.where(context[20:27] >= 1)
    else:
      indices = np.where(context[0:7] >= 1)
      
    return indices[0][0]
  

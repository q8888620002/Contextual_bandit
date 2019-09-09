
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
    print(context.shape)
    
    print(context[: 20:26])
    return np.random.choice(range(self.hparams.num_actions))
  

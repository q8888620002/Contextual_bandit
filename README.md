# Towards uncertainty aware treatment assignments inprecision oncology. 
# An evolving contextual banditproblem

This is the implementation of treatment assignment with *[Deep Bayesian Bandits](https://arxiv.org/abs/1802.09127)* paper, published in [ICLR](https://iclr.cc/) 2018. We provide a benchmark to test decision-making
algorithms for contextual-bandits. In particular, the current library implements
a variety of algorithms (many of them based on approximate Bayesian Neural
Networks and Thompson sampling), and a number of real and syntethic data
problems exhibiting a diverse set of properties.

It is a Python library that uses [TensorFlow](https://www.tensorflow.org/).

**Contact**. This repository is maintained by [Mingyu Lu]. Feel free to reach out directly at [mingyulu@mit.edu](mailto:mingyulu@mit.edu) with any questions or comments.


We first briefly introduce contextual bandits, Thompson sampling, enumerate the
implemented algorithms, and the available data sources. Then, we provide a
simple complete example illustrating how to use the library.

## Contextual Bandits

Contextual bandits are a rich decision-making framework where an algorithm has
to choose among a set of *k* actions at every time step *t*, after observing
a context (or side-information) denoted by *X<sub>t</sub>*. The general pseudocode for
the process if we use algorithm **A** is as follows:

```
At time t = 1, ..., T:
  1. Observe new context: X_t
  2. Choose action: a_t = A.action(X_t)
  3. Observe reward: r_t
  4. Update internal state of the algorithm: A.update((X_t, a_t, r_t))
```

The goal is to maximize the total sum of rewards: &sum;<sub>t</sub> r<sub>t</sub>

For example, each *X<sub>t</sub>* could encode the properties of a specific user (and
the time or day), and we may have to choose an ad, discount coupon, treatment,
hyper-parameters, or version of a website to show or provide to the user.
Hopefully, over time, we will learn how to match each type of user to the most
beneficial personalized action under some metric (the reward).

## Data


## Usage: Basic Example

This library requires Tensorflow, Numpy, Pandas, and Deep Bayesian Bandits.

The file *treatment_assignment.py* provides a complete example on how to use the
library. We run the code:

```
    python treatment_assignment.py
```

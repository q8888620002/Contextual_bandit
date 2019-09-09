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

## Thompson Sampling

Thompson Sampling is a meta-algorithm that chooses an action for the contextual
bandit in a statistically efficient manner, simultaneously finding the best arm
while attempting to incur low cost. Informally speaking, we assume the expected
reward is given by some function
**E**[r<sub>t</sub> | X<sub>t</sub>, a<sub>t</sub>] = f(X<sub>t</sub>, a<sub>t</sub>).
Unfortunately, function **f** is unknown, as otherwise we could just choose the
action with highest expected value:
a<sub>t</sub><sup>*</sup> = arg max<sub>i</sub> f(X<sub>t</sub>, a<sub>t</sub>).

The idea behind Thompson Sampling is based on keeping a posterior distribution
&pi;<sub>t</sub> over functions in some family f &isin; F after observing the first
*t-1* datapoints. Then, at time *t*, we sample one potential explanation of
the underlying process: f<sub>t</sub> &sim; &pi;<sub>t</sub>, and act optimally (i.e., greedily)
*according to f<sub>t</sub>*. In other words, we choose
a<sub>t</sub> = arg max<sub>i</sub> f<sub>t</sub>(X<sub>t</sub>, a<sub>i</sub>).
Finally, we update our posterior distribution with the new collected
datapoint (X<sub>t</sub>, a<sub>t</sub>, r<sub>t</sub>).

The main issue is that keeping an updated posterior &pi;<sub>t</sub> (or, even,
sampling from it) is often intractable for highly parameterized models like deep
neural networks. The algorithms we list in the next section provide tractable
*approximations* that can be used in combination with Thompson Sampling to solve
the contextual bandit problem.

In the code snippet at the bottom, we show how to instantiate some of these
algorithms, and how to run the contextual bandit simulator, and display the
high-level results.

## Data

In the paper we use two types of contextual datasets: synthetic and based on
real-world data.

We provide functions that sample problems from those datasets. In the case of
real-world data, you first need to download the raw datasets, and pass the route
to the functions. Links for the datasets are provided below.

### Synthetic Datasets

Synthetic datasets are contained in the *synthetic_data_sampler.py* file. In
particular, it includes:

1.  **Linear data**. Provides a number of linear arms, and Gaussian contexts.

2.  **Sparse linear data**. Provides a number of sparse linear arms, and
    Gaussian contexts.

3.  **Wheel bandit data**. Provides sampled data from the wheel bandit data, see
    [Section 5.4](https://arxiv.org/abs/1802.09127) in the paper.

### Real-World Datasets

Real-world data generating functions are contained in the *data_sampler.py*
file.

In particular, it includes:

1.  **Mushroom data**. Each incoming context represents a different type of
    mushroom, and the actions are eat or no-eat. Eating an edible mushroom
    provides positive reward, while eating a poisonous one provides positive
    reward with probability *p*, and a large negative reward with probability
    *1-p*. All the rewards, and the value of *p* are customizable. The
    [dataset](https://archive.ics.uci.edu/ml/datasets/mushroom) is part of the
    UCI repository, and the bandit problem was proposed in Blundell et al.
    (2015). Data is available [here](https://storage.googleapis.com/bandits_datasets/mushroom.data)
    or alternatively [here](https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/),
    use the *agaricus-lepiota.data* file.

2.  **Stock data**. We created the Financial Dataset by pulling the stock prices
    of *d = 21* publicly traded companies in NYSE and Nasdaq, for the last 14
    years (*n = 3713*). For each day, the context was the price difference
    between the beginning and end of the session for each stock. We
    synthetically created the arms to be a linear combination of the contexts,
    representing *k = 8* different potential portfolios. Data is available
    [here](https://storage.googleapis.com/bandits_datasets/raw_stock_contexts).

3.  **Jester data**. We create a recommendation system bandit problem as
    follows. The Jester Dataset (Goldberg et al., 2001) provides continuous
    ratings in *[-10, 10]* for 100 jokes from a total of 73421 users. We find
    a *complete* subset of *n = 19181* users rating all 40 jokes. Following
    Riquelme et al. (2017), we take *d = 32* of the ratings as the context of
    the user, and *k = 8* as the arms. The agent recommends one joke, and
    obtains the reward corresponding to the rating of the user for the selected
    joke. Data is available [here](https://storage.googleapis.com/bandits_datasets/jester_data_40jokes_19181users.npy).

4.  **Statlog data**. The Shuttle Statlog Dataset (Asuncion & Newman, 2007)
    provides the value of *d = 9* indicators during a space shuttle flight,
    and the goal is to predict the state of the radiator subsystem of the
    shuttle. There are *k = 7* possible states, and if the agent selects the
    right state, then reward 1 is generated. Otherwise, the agent obtains no
    reward (*r = 0*). The most interesting aspect of the dataset is that one
    action is the optimal one in 80% of the cases, and some algorithms may
    commit to this action instead of further exploring. In this case, the number
    of contexts is *n = 43500*. Data is available [here](https://storage.googleapis.com/bandits_datasets/shuttle.trn) or alternatively
    [here](https://archive.ics.uci.edu/ml/datasets/Statlog+\(Shuttle\)), use
    *shuttle.trn* file.

5.  **Adult data**. The Adult Dataset (Kohavi, 1996; Asuncion & Newman, 2007)
    comprises personal information from the US Census Bureau database, and the
    standard prediction task is to determine if a person makes over 50K a year
    or not. However, we consider the *k = 14* different occupations as
    feasible actions, based on *d = 94* covariates (many of them binarized).
    As in previous datasets, the agent obtains a reward of 1 for making the
    right prediction, and 0 otherwise. The total number of contexts is *n =
    45222*. Data is available [here](https://storage.googleapis.com/bandits_datasets/adult.full) or alternatively
    [here](https://archive.ics.uci.edu/ml/datasets/adult), use *adult.data*
    file.

6.  **Census data**. The US Census (1990) Dataset (Asuncion & Newman, 2007)
    contains a number of personal features (age, native language, education...)
    which we summarize in *d = 389* covariates, including binary dummy
    variables for categorical features. Our goal again is to predict the
    occupation of the individual among *k = 9* classes. The agent obtains
    reward 1 for making the right prediction, and 0 otherwise. Data is available
    [here](https://storage.googleapis.com/bandits_datasets/USCensus1990.data.txt) or alternatively [here](https://archive.ics.uci.edu/ml/datasets/US+Census+Data+\(1990\)), use
    *USCensus1990.data.txt* file.

7.  **Covertype data**. The Covertype Dataset (Asuncion & Newman, 2007)
    classifies the cover type of northern Colorado forest areas in *k = 7*
    classes, based on *d = 54* features, including elevation, slope, aspect,
    and soil type. Again, the agent obtains reward 1 if the correct class is
    selected, and 0 otherwise. Data is available [here](https://storage.googleapis.com/bandits_datasets/covtype.data) or alternatively
    [here](https://archive.ics.uci.edu/ml/datasets/covertype), use
    *covtype.data* file.

In datasets 4-7, each feature of the dataset is normalized first.

## Usage: Basic Example

This library requires Tensorflow, Numpy, Pandas, and Deep Bayesian Bandits.

The file *treatment_assignment.py* provides a complete example on how to use the
library. We run the code:

```
    python treatment_assignment.py
```

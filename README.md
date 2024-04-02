# PAI-ETHZ
This repository contains the projects completed for the Probabilistic Artificial Intelligence course at ETHZ for the Fall Semester 2023.

## Task 1

This task implements a _Gaussian Process Regression_ model to predict air pollution at locations in a city which are lacking measurements, to determine suitable residential areas with low air pollution.

## Task 2

The goal of this task is to perform _classification_ on satellite images to determine the type of land usage. While the training data has well-defined samples, the test data may contain some ambiguous images, that is, they cannot be assigned to one particular label. For example, there might be mixed land usage such as a lake surrounded by a forest. The importance of this task lies in the _quantification of the classification uncertainty_, i.e., being able to distinguish between an image certainly belonging to a class and one that might not be assigned to a class with high certainty. To achieve a well-calibrated model, we implemented a _Bayesian Neural Network_ architecture and used _approximate inference_ to estimate the intractable posterior distribution. The technique used here for approximation is the _Stochastic Weight Averaging Gaussian_ [Maddox et al., 2019](https://arxiv.org/abs/1902.02476), which stores weight statistics during training, and uses them to fit an approximate Gaussian posterior.

## Task 3

The goal of this project is to tune the features of a function which has a high evaluation cost, using _Bayesian Optimization_, with limited resources. In this case, the problem description is a drug candidate which should be bioavailable enough to reach its intended target and easy to synthesize, with the objective being a proxy for bioavailability. This task implements the _Expected Improvement_ acquisition function for Bayesian Optimization, with a small tweak, i.e., restricting the choice of parameters to those that are safe with high probability.

## Task 4

We develop and train a _Reinforcement Learning_ agent which will swing up an _inverted pendulum_ from a downward position to an upward position and try to hold it there, using the off-policy _Soft Actor-Critic_ algorithm. This is done in a simulator setting, i.e., the RL agent interacts with the pendulum a finite number of times, and thus learns a control policy.

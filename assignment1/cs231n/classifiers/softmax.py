import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_train = X.shape[0]
  num_classes = W.shape[1]

  for i in range(num_train):
    scores = X[i].dot(W)
    exp_scores = np.exp(scores - np.max(scores))
    ps = exp_scores / np.sum(exp_scores)
    p1 = ps[y[i]]
    loss += - np.log(p1)

    for j in range(num_classes):
      p2 = ps[j]
      dW[:, j] += (-1 / p1) * ((j == y[i])*p1 - p1 * p2) * X[i]

  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * 2 * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_train = X.shape[0]
  num_classes = W.shape[1]

  scores = X.dot(W)
  ss = scores - np.max(scores, axis=1)[:, np.newaxis]
  exp_scores = np.exp(ss)
  ps2 = exp_scores / np.sum(exp_scores, axis=1)[:, np.newaxis]
  ys = ps2[range(ps2.shape[0]), y]
  loss += - np.sum(np.log(ys))

  pp = - ps2 * ys[:, np.newaxis]
  pp[range(pp.shape[0]), y] += ys
  dsm = (-1 / ys[:, np.newaxis]) * pp
  dW = X.T.dot(dsm)

  # for i in range(num_train):
  #   scores = X[i].dot(W)
  #   exp_scores = np.exp(scores - np.max(scores))
  #   ps = ps2[i]
  #   p1 = ps[y[i]]
  #   loss += - np.log(p1)
  #
  #   for j in range(num_classes):
  #     p2 = ps[j]
  #     dW[:, j] += (-1 / p1) * ((j == y[i])*p1 - p1 * p2) * X[i]
  #
  #   pp = - p1 * ps
  #   pp[y[i]] += p1
  #   dW += (-1 / p1) * pp * X[i][:, np.newaxis]

  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * 2 * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


import tensorflow as tf
import os
import numpy as np
from machine_learning_models.utils import *


def log_reg_model_fn(features, targets, mode, params):

        # model definition - very simple in the case of logistic regression
        logits = tf.layers.dense(features, units=params.target_size)
        predictions = tf.nn.softmax(logits)

        # Estimator creation - depending on what mode is called, we return the equivalent estimator eg. PREDICT, TRAIN
        # or EVAL

        # PREDICT estimator creation
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"predictions": predictions})

        # Cost function - Cross entropy / negative log likelihood
        cost = tf.losses.softmax_cross_entropy(targets, logits)

        # Set evaluation metrics
        accuracy = tf.metrics.accuracy(labels=targets, predictions=predictions)
        eval_metrics = {'accuracy' : accuracy}
        tf.summary.scalar('accuracy', accuracy[1]) # Used for tensorboard

        # EVAL estimator creation
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=cost,
                eval_metric_ops=eval_metrics
            )

        # Optimization - Adaptive momentum estimation eg. Adam
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        train_operation = optimizer.minimize(cost, global_step=tf.train.get_global_step())

        # TRAIN estimator creation
        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=cost,
                eval_metric_ops=eval_metrics,
                train_op=train_operation
            )

def main():

    train_x = np.zeros(10, 10)
    train_y = np.zeros(10)

    eval_x = np.zeros(10, 10)
    eval_y = np.zeros(10)


    params = AttrDict()
    params.input_size = 25  # todo change to actual input size
    params.target_size = 4
    params.learning_rate = 0.001
    params.training_epochs = 50

    # Create estimator model
    nn = tf.estimator.Estimator(model_fn=log_reg_model_fn, params=params, model_dir=os.getcwd())

    # Create training input
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=train_x,
        y=train_y,
        num_epochs=None,
        shuffle=True)

    # Train model
    nn.train(train_input_fn, steps=params.training_epochs)

    # Create evaluate input
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=eval_x,
        y=eval_y,
        num_epochs=1,
        shuffle=False
    )

    # Evalaute model
    evaluation = nn.evaluate(eval_input_fn)
    print("Eval Cost: {}".format(evaluation['loss']))
    print("Eval Accuracy: {}".format(evaluation['accuracy']))



if __name__ == "__main__":
    tf.app.run(main=main)


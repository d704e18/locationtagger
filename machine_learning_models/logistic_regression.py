import tensorflow as tf
import os
from machine_learning_models.utils import *

tf.logging.set_verbosity(tf.logging.INFO)


def log_reg_model_fn(features, labels, mode, params):

        # model definition - very simple in the case of logistic regression
        logits = tf.layers.dense(features, units=params['target_size'])
        predictions = tf.nn.softmax(logits)

        # Estimator creation - depending on what mode is called, we return the equivalent estimator eg. PREDICT, TRAIN
        # or EVAL

        # PREDICT estimator creation
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"predictions": predictions})

        # Cost function - Cross entropy / negative log likelihood
        cost = tf.losses.softmax_cross_entropy(labels, logits)

        # Set evaluation metrics
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)
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
        optimizer = tf.train.AdamOptimizer(params['learning_rate'])
        train_operation = optimizer.minimize(cost, global_step=tf.train.get_global_step())

        # TRAIN estimator creation
        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=cost,
                eval_metric_ops=eval_metrics,
                train_op=train_operation
            )

def main(unused_arg):

    path = os.getcwd()
    parent = '/'.join(path.split('/')[:-1])

    train_x, train_y, validation_x, validation_y, test_x, test_y = load_data(parent+"/data/whole_week.pkl")

    model_params = {'input_size' : train_x.shape[1],
                    'target_size' : train_y.shape[1],
                    'learning_rate' : 0.001,
                    'training_epochs' : 10000
                    }
    i = 2
    # Create estimator model
    nn = tf.estimator.Estimator(model_fn=log_reg_model_fn, params=model_params, model_dir="{}/trained_models/{}".format(parent, i))

    # Create training input
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=train_x,
        y=train_y,
        num_epochs=None,
        shuffle=True)

    # Train model
    nn.train(train_input_fn, steps=model_params['training_epochs'])

    # Create evaluate input
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=test_x,
        y=test_y,
        num_epochs=1,
        shuffle=False
    )

    # Evalaute model
    evaluation = nn.evaluate(eval_input_fn)
    print("Eval Cost: {}".format(evaluation['loss']))
    print("Eval Accuracy: {}".format(evaluation['accuracy']))



if __name__ == "__main__":
    tf.app.run(main=main)


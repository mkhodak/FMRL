"""
Train a model on miniImageNet.
"""

import random

import tensorflow as tf

from supervised_reptile.args import argument_parser, model_kwargs, train_kwargs, evaluate_kwargs
from supervised_reptile.eval import evaluate
from supervised_reptile.models import MiniImageNetModel
from supervised_reptile.miniimagenet import read_dataset
from supervised_reptile.train import train

DATA_DIR = 'data/miniimagenet'

def main():
    """
    Load data and train a model on it.
    """
    args = argument_parser().parse_args()
    random.seed(args.seed)

    train_set, val_set, test_set = read_dataset(DATA_DIR)
    model = MiniImageNetModel(args.classes, **model_kwargs(args))

    with tf.Session() as sess:
        for i in range(5):

            if not args.pretrained:
                print('Training...')
                train(sess, model, train_set, test_set, args.checkpoint+str(i), **train_kwargs(args))
            else:
                print('Restoring from checkpoint...')
                tf.train.Saver().restore(sess, tf.train.latest_checkpoint(args.checkpoint))

            print('Evaluating...')
            eval_kwargs = evaluate_kwargs(args)
            #print('Train accuracy: ' + str(evaluate(sess, model, train_set, **eval_kwargs)))
            #print('Validation accuracy: ' + str(evaluate(sess, model, val_set, **eval_kwargs)))
            print('Test accuracy: ' + str(evaluate(sess, model, test_set, **eval_kwargs)))

if __name__ == '__main__':
    main()

#python -u eval_miniimagenet.py --shots 1 --inner-batch 10 --inner-iters 40 --meta-step 1 --meta-batch 5 --meta-iters 20000 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 15 --checkpoint ckpt_minimany15 --transductive --eval-samples 1000 --ftrl 1.0 2>&1 | tee minimany_eval15.log
#
#python -u eval_miniimagenet.py --inner-batch 10 --inner-iters 40 --meta-step 1 --meta-batch 5 --meta-iters 20000 --eval-batch 15 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 15 --checkpoint ckpt_minimany55 --transductive --eval-samples 1000 --ftrl 1.0 2>&1 | tee minimany_eval55.log

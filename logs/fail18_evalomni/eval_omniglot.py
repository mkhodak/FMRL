"""
Train a model on Omniglot.
"""

import random

import tensorflow as tf

from supervised_reptile.args import argument_parser, model_kwargs, train_kwargs, evaluate_kwargs
from supervised_reptile.eval import evaluate
from supervised_reptile.models import OmniglotModel
from supervised_reptile.omniglot import read_dataset, split_dataset, augment_dataset
from supervised_reptile.train import train

DATA_DIR = 'data/omniglot'

def main():
    """
    Load data and train a model on it.
    """
    args = argument_parser().parse_args()
    random.seed(args.seed)

    train_set, test_set = split_dataset(read_dataset(DATA_DIR))
    train_set = list(augment_dataset(train_set))
    test_set = list(test_set)

    model = OmniglotModel(args.classes, **model_kwargs(args))

    with tf.Session() as sess:
        for i in range(10):

            if not args.pretrained:
                print('Training...')
                train(sess, model, train_set, test_set, args.checkpoint+str(i), **train_kwargs(args))
            else:
                print('Restoring from checkpoint...')
                tf.train.Saver().restore(sess, tf.train.latest_checkpoint(args.checkpoint))

            print('Evaluating...')
            eval_kwargs = evaluate_kwargs(args)
            #print('Train accuracy: ' + str(evaluate(sess, model, train_set, **eval_kwargs)))
            print('Test accuracy: ' + str(evaluate(sess, model, test_set, **eval_kwargs)))

if __name__ == '__main__':
    main()

#python -u eval_omniglot.py --shots 1 --inner-batch 10 --inner-iters 25 --meta-step 1 --meta-batch 5 --meta-iters 20000 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 10 --checkpoint ckpt_omnimany_eval15 --transductive --eval-samples 1000 --ftrl 1.0 2>&1 | tee omnimany_eval15.log
#python -u eval_omniglot.py --train-shots 10 --inner-batch 10 --inner-iters 25 --learning-rate 0.001 --meta-step 1 --meta-step-final 0 --meta-batch 5 --meta-iters 20000 --eval-batch 5 --eval-iters 50 --checkpoint ckpt_omnimany_eval55 --transductive --eval-samples 1000 --ftrl 1.0 2>&1 | tee omnimany_eval55.log

#python -u eval_omniglot.py --shots 1 --classes 20 --inner-batch 20 --inner-iters 50 --meta-step 1 --meta-batch 5 --meta-iters 20000 --eval-batch 10 --eval-iters 50 --learning-rate 0.0005 --meta-step-final 0 --train-shots 10 --checkpoint ckpt_omnimany_eval120 --transductive --eval-samples 1000 --ftrl 1.0 2>&1 | tee omnimany_eval120.log
#python -u eval_omniglot.py --classes 20 --inner-batch 20 --inner-iters 50 --meta-step 1 --meta-batch 5 --meta-iters 20000 --eval-batch 10 --eval-iters 50 --learning-rate 0.0005 --meta-step-final 0 --train-shots 10 --checkpoint ckpt_omnimany_eval520 --transductive --eval-samples 1000 --ftrl 1.0 2>&1 | tee omnimany_eval520.log

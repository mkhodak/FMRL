python -u run_miniimagenet.py --shots 1 --inner-batch 10 --inner-iters 8 --meta-step 1 --meta-batch 5 --meta-iters 100000 --eval-batch 5 --eval-iters 50 --learning-rate 0.01 --meta-step-final 0 --train-shots 15 --checkpoint ckpt_mini15 --transductive --ftrl 0.1 --sgd 2>&1 | tee mini15.log

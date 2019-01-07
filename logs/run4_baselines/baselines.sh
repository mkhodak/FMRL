python -u run_omniglot.py --shots 1 --inner-batch 10 --inner-iters 5 --meta-step 1 --meta-batch 5 --meta-iters 25000 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 10 --checkpoint ckpt_omni_baseline --transductive 2>&1 | tee omni_baseline.log

python -u run_miniimagenet.py --shots 1 --inner-batch 10 --inner-iters 8 --meta-step 1 --meta-batch 5 --meta-iters 10000 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 15 --checkpoint ckpt_mini_baseline --transductive 2>&1 | tee mini_baseline.log



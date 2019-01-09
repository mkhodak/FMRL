python -u run_omniglot.py --train-shots 10 --inner-batch 10 --inner-iters 5 --learning-rate 0.01 --meta-step 1 --meta-step-final 0 --meta-batch 5 --meta-iters 100000 --eval-batch 5 --eval-iters 50 --checkpoint ckpt_omni55 --transductive --ftrl 1.0 --sgd 2>&1 | tee omni55.log

python -u run_omniglot.py --classes 20 --inner-batch 20 --inner-iters 10 --meta-step 1 --meta-batch 5 --meta-iters 200000 --eval-batch 10 --eval-iters 50 --learning-rate 0.005 --meta-step-final 0 --train-shots 10 --checkpoint ckpt_omni520 --transductive --ftrl 1.0 --sgd 2>&1 | tee omni520.log

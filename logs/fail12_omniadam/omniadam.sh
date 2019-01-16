python -u run_omniglot.py --shots 1 --inner-batch 10 --inner-iters 5 --meta-step 1 --meta-batch 5 --meta-iters 100000 --eval-batch 5 --eval-iters 50 --learning-rate 0.01 --meta-step-final 0 --train-shots 10 --checkpoint ckpt_omniadam15 --transductive --ftrl 0.1 2>&1 | tee omniadam15.log

python -u run_omniglot.py --shots 1 --classes 20 --inner-batch 20 --inner-iters 10 --meta-step 1 --meta-batch 5 --meta-iters 200000 --eval-batch 10 --eval-iters 50 --learning-rate 0.005 --meta-step-final 0 --train-shots 10 --checkpoint ckpt_omniadam120 --transductive --ftrl 0.1 2>&1 | tee omniadam120.log

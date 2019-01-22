python -u eval_omniglot.py --train-shots 10 --inner-batch 10 --inner-iters 5 --learning-rate 0.001 --meta-step 1 --meta-step-final 0 --meta-batch 5 --meta-iters 200 --eval-batch 5 --eval-iters 50 --checkpoint ckpt_omni_rept --eval-samples 500 2>&1 | tee omni_rept.log

python -u eval_omniglot.py --train-shots 10 --inner-batch 10 --inner-iters 5 --learning-rate 0.001 --meta-step 1 --meta-step-final 0 --meta-batch 5 --meta-iters 200 --eval-batch 5 --eval-iters 50 --checkpoint ckpt_omni_ftrl --eval-samples 500 --ftrl 0.01 2>&1 | tee omni_ftrl.log

python -u eval_omniglot.py --train-shots 10 --inner-batch 10 --inner-iters 25 --learning-rate 0.001 --meta-step 1 --meta-step-final 0 --meta-batch 5 --meta-iters 200 --eval-batch 5 --eval-iters 50 --checkpoint ckpt_omni_many --eval-samples 500 2>&1 | tee omni_many.log

python -u eval_omniglot.py --train-shots 10 --inner-batch 10 --inner-iters 25 --learning-rate 0.001 --meta-step 1 --meta-step-final 0 --meta-batch 5 --meta-iters 200 --eval-batch 5 --eval-iters 50 --checkpoint ckpt_omni_both --eval-samples 500 --ftrl 1.0 2>&1 | tee omni_both.log

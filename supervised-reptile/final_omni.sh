python -u final_omniglot.py --train-shots 10 --inner-batch 10 --inner-iters 5 --learning-rate 0.001 --meta-step 1 --meta-step-final 0 --meta-batch 5 --meta-iters 5001 --eval-batch 5 --eval-iters 50 --checkpoint ckpt_omni_rept --eval-samples 500 --eval-interval 250 2>&1 | tee omni_rept.log

python -u final_omniglot.py --train-shots 10 --inner-batch 10 --inner-iters 5 --learning-rate 0.001 --meta-step 1 --meta-step-final 0 --meta-batch 5 --meta-iters 5001 --eval-batch 5 --eval-iters 50 --checkpoint ckpt_omni_ftrl --eval-samples 500 --eval-interval 250 --ftrl 1.0 2>&1 | tee omni_ftrl.log

python -u final_omniglot.py --train-shots 10 --inner-batch 10 --inner-iters 25 --learning-rate 0.001 --meta-step 1 --meta-step-final 0 --meta-batch 5 --meta-iters 5001 --eval-batch 5 --eval-iters 50 --checkpoint ckpt_omni_many --eval-samples 500 --eval-interval 250 2>&1 | tee omni_many.log

python -u final_omniglot.py --train-shots 10 --inner-batch 10 --inner-iters 25 --learning-rate 0.001 --meta-step 1 --meta-step-final 0 --meta-batch 5 --meta-iters 5001 --eval-batch 5 --eval-iters 50 --checkpoint ckpt_omni_both --eval-samples 500 --eval-interval 250 --ftrl 1.0 2>&1 | tee omni_both.log

python -u final_omniglot.py --shots 1 --inner-batch 10 --inner-iters 5 --meta-step 1 --meta-batch 5 --meta-iters 5001 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 10 --checkpoint ckpt_omni1shot_rept --eval-samples 500 --eval-interval 250 2>&1 | tee omni1shot_rept.log

python -u final_omniglot.py --shots 1 --inner-batch 10 --inner-iters 5 --meta-step 1 --meta-batch 5 --meta-iters 5001 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 10 --checkpoint ckpt_omni1shot_ftrl --eval-samples 500 --eval-interval 250 --ftrl 1.0 2>&1 | tee omni1shot_ftrl.log

python -u final_omniglot.py --shots 1 --inner-batch 10 --inner-iters 25 --meta-step 1 --meta-batch 5 --meta-iters 5001 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 10 --checkpoint ckpt_omni1shot_many --eval-samples 500 --eval-interval 250 2>&1 | tee omni1shot_many.log

python -u final_omniglot.py --shots 1 --inner-batch 10 --inner-iters 25 --meta-step 1 --meta-batch 5 --meta-iters 5001 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 10 --checkpoint ckpt_omni1shot_both --eval-samples 500 --eval-interval 250 --ftrl 1.0 2>&1 | tee omni1shot_both.log

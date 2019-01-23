python -u final_miniimagenet.py --inner-batch 10 --inner-iters 8 --meta-step 1 --meta-batch 5 --meta-iters 5001 --eval-batch 15 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 15 --eval-samples 500 --eval-interval 250 --checkpoint ckpt_mini_rept 2>&1 | tee mini_rept.log

python -u final_miniimagenet.py --inner-batch 10 --inner-iters 8 --meta-step 1 --meta-batch 5 --meta-iters 5001 --eval-batch 15 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 15 --eval-samples 500 --eval-interval 250 --ftrl 1.0 --checkpoint ckpt_mini_ftrl 2>&1 | tee mini_ftrl.log

python -u final_miniimagenet.py --inner-batch 10 --inner-iters 40 --meta-step 1 --meta-batch 5 --meta-iters 5001 --eval-batch 15 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 15 --eval-samples 500 --eval-interval 250 --checkpoint ckpt_mini_many 2>&1 | tee mini_many.log

python -u final_miniimagenet.py --inner-batch 10 --inner-iters 40 --meta-step 1 --meta-batch 5 --meta-iters 5001 --eval-batch 15 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 15 --eval-samples 500 --eval-interval 250 --ftrl 1.0 --checkpoint ckpt_mini_both 2>&1 | tee mini_both.log

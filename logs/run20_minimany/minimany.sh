for weight in 0.01 0.1 1 ; do
    #python -u run_miniimagenet.py --shots 1 --inner-batch 10 --inner-iters 40 --meta-step 1 --meta-batch 5 --meta-iters 10000 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 15 --checkpoint ckpt_minimany$weight --transductive --eval-samples 1000 --ftrl $weight 2>&1 | tee minimany$weight.log
    python -u run_miniimagenet.py --inner-batch 10 --inner-iters 40 --meta-step 1 --meta-batch 5 --meta-iters 10000 --eval-batch 15 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 15 --checkpoint ckpt_minimany5shot$weight --transductive --eval-samples 1000 --ftrl $weight 2>&1 | tee minimany5shot$weight.log
done

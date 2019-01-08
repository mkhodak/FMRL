for weight in 0.01 0.5; do
    python -u run_miniimagenet.py --shots 1 --inner-batch 10 --inner-iters 8 --meta-step 1 --meta-batch 5 --meta-iters 10000 --eval-batch 5 --eval-iters 50 --learning-rate 0.01 --meta-step-final 0 --train-shots 15 --checkpoint ckpt_mini$weight --transductive --eval-samples 1000 --ftrl $weight --sgd 2>&1 | tee mini$weight.log
done
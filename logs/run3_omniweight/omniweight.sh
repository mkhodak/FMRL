for weight in 0.25 0.5 2 4; do
    python -u run_omniglot.py --shots 1 --inner-batch 10 --inner-iters 5 --meta-step 1 --meta-batch 5 --meta-iters 25000 --eval-batch 5 --eval-iters 50 --learning-rate 0.01 --meta-step-final 0 --train-shots 10 --checkpoint ckpt_omni$weight --transductive --eval-samples 1000 --ftrl $weight --sgd 2>&1 | tee omni$weight.log
done
OPTIONS="--transductive --eval-samples 200 --eval-interval 1000"

for ITER in 4 8 16 32 ; do
    python -u run_miniimagenet.py --inner-batch 10 --inner-iters $ITER --meta-step 1 --meta-batch 5 --meta-iters 20000 --eval-batch 15 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 15 --checkpoint ckpt_m55_i$ITER $OPTIONS 2>&1 | tee m55_i$ITER.log
    python -u run_miniimagenet.py --shots 1 --inner-batch 10 --inner-iters $ITER --meta-step 1 --meta-batch 5 --meta-iters 20000 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 15 --checkpoint ckpt_m15_i$ITER $OPTIONS 2>&1 | tee m15_i$ITER.log
done

for SHOT in 5 10 20 25 ; do
    python -u run_miniimagenet.py --inner-batch 10 --inner-iters 8 --meta-step 1 --meta-batch 5 --meta-iters 20000 --eval-batch 15 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots $SHOT --checkpoint ckpt_m55_s$SHOT $OPTIONS 2>&1 | tee m55_s$SHOT.log
    python -u run_miniimagenet.py --shots 1 --inner-batch 10 --inner-iters 8 --meta-step 1 --meta-batch 5 --meta-iters 20000 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots $SHOT --checkpoint ckpt_m15_s$SHOT $OPTIONS 2>&1 | tee m15_s$SHOT.log
done

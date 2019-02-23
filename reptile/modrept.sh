OPTIONS="--transductive --eval-samples 200 --eval-interval 1000"

for ITER in 8 16 24 32 ; do
    python -u run_miniimagenet.py --inner-batch 10 --inner-iters $ITER --meta-step 1 --meta-batch 5 --meta-iters 20000 --eval-batch 15 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 15 --checkpoint ckpt_m55_i$ITER $OPTIONS 2>&1 | tee m55_i$ITER.log
done
for ITER in 5 10 15 20 ; do
    python -u run_omniglot.py --train-shots 10 --inner-batch 10 --inner-iters $ITER --learning-rate 0.001 --meta-step 1 --meta-step-final 0 --meta-batch 5 --meta-iters 20000 --eval-batch 5 --eval-iters 50 --checkpoint ckpt_o55_i$ITER $OPTIONS 2>&1 | tee o55_i$ITER.log
    python -u run_omniglot.py --classes 20 --inner-batch 20 --inner-iters $ITER --meta-step 1 --meta-batch 5 --meta-iters 20000 --eval-batch 10 --eval-iters 50 --learning-rate 0.0005 --meta-step-final 0 --train-shots 10 --checkpoint ckpt_o520_i$ITER $OPTIONS 2>&1 | tee o520_i$ITER.log
done

for SHOT in 5 10 15 20 ; do
    python -u run_miniimagenet.py --inner-batch 10 --inner-iters 8 --meta-step 1 --meta-batch 5 --meta-iters 20000 --eval-batch 15 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots $SHOT --checkpoint ckpt_m55_s$SHOT $OPTIONS 2>&1 | tee m55_s$SHOT.log
    python -u run_omniglot.py --train-shots $SHOT --inner-batch 10 --inner-iters 5 --learning-rate 0.001 --meta-step 1 --meta-step-final 0 --meta-batch 5 --meta-iters 20000 --eval-batch 5 --eval-iters 50 --checkpoint ckpt_o55_s$SHOT $OPTIONS 2>&1 | tee o55_s$SHOT.log
    python -u run_omniglot.py --classes 20 --inner-batch 20 --inner-iters 10 --meta-step 1 --meta-batch 5 --meta-iters 20000 --eval-batch 10 --eval-iters 50 --learning-rate 0.0005 --meta-step-final 0 --train-shots $SHOT --checkpoint ckpt_o520_s$SHOT $OPTIONS 2>&1 | tee o520_s$SHOT.log
done

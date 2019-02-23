OPTIONS='--transductive --eval-interval 10000 --eval-samples 1000'

for ITER in 8 16 24 32 40 ; do

    python sweep_miniimagenet.py --shots 1 --inner-batch 10 --inner-iters $ITER --meta-step 1 --meta-batch 5 --meta-iters 50000 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 15 --checkpoint ckpt$ITER'_m15sweep' $OPTIONS 2>&1 | tee iter$ITER'_m15sweep.log'

    python sweep_miniimagenet.py --inner-batch 10 --inner-iters $ITER --meta-step 1 --meta-batch 5 --meta-iters 50000 --eval-batch 15 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 15 --checkpoint ckpt$ITER'_m55sweep' $OPTIONS 2>&1 | tee iter$ITER'_m55sweep.log'

done

OPTIONS='--transductive --eval-interval 10000 --eval-samples 1000 --ftrl'

for COEF in 0.01 0.1 1.0 10.0 100.0 ; do

    python sweep_miniimagenet.py --shots 1 --inner-batch 10 --inner-iters 8 --meta-step 1 --meta-batch 5 --meta-iters 50000 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 15 --checkpoint ckpt$COEF'_m15sweep' $OPTIONS $COEF 2>&1 | tee reg$COEF'_m15sweep.log'

    python sweep_miniimagenet.py --inner-batch 10 --inner-iters 8 --meta-step 1 --meta-batch 5 --meta-iters 50000 --eval-batch 15 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 15 --checkpoint ckpt$COEF'_m55sweep' $OPTIONS $COEF 2>&1 | tee reg$COEF'_m55sweep.log'

done

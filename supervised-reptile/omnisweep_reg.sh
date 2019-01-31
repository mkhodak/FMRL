OPTIONS='--transductive --eval-interval 10000 --eval-samples 1000 --ftrl'

for COEF in 0.01 0.1 1.0 10.0 100.0 ; do

    python sweep_omniglot.py --shots 1 --inner-batch 10 --inner-iters 5 --meta-step 1 --meta-batch 5 --meta-iters 50000 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 10 --checkpoint ckpt$COEF'_o15sweep' $OPTIONS $COEF 2>&1 | tee reg$COEF'_o15sweep.log'

    python sweep_omniglot.py --train-shots 10 --inner-batch 10 --inner-iters 5 --learning-rate 0.001 --meta-step 1 --meta-step-final 0 --meta-batch 5 --meta-iters 50000 --eval-batch 5 --eval-iters 50 --checkpoint ckpt$COEF'_o55sweep' $OPTIONS $COEF 2>&1 | tee reg$COEF'_o55sweep.log'

    python sweep_omniglot.py --shots 1 --classes 20 --inner-batch 20 --inner-iters 10 --meta-step 1 --meta-batch 5 --meta-iters 50000 --eval-batch 10 --eval-iters 50 --learning-rate 0.0005 --meta-step-final 0 --train-shots 10 --checkpoint ckpt$COEF'_o120sweep' $OPTIONS $COEF 2>&1 | tee reg$COEF'_o120sweep.log' 

    python sweep_omniglot.py --classes 20 --inner-batch 20 --inner-iters 10 --meta-step 1 --meta-batch 5 --meta-iters 50000 --eval-batch 10 --eval-iters 50 --learning-rate 0.0005 --meta-step-final 0 --train-shots 10 --checkpoint ckpt$COEF'_o520sweep' $OPTIONS $COEF 2>&1 | tee reg$COEF'_o520sweep.log'

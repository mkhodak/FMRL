OPTIONS='--transductive --eval-interval 10000 --eval-samples 1000'

for ITER in 5 10 25 50 ; do

    python sweep_omniglot.py --shots 1 --inner-batch 10 --inner-iters $ITER --meta-step 1 --meta-batch 5 --meta-iters 50000 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 10 --checkpoint ckpt$ITER'_o15sweep' $OPTIONS 2>&1 | tee iter$ITER'_o15sweep.log'

    python sweep_omniglot.py --train-shots 10 --inner-batch 10 --inner-iters $ITER --learning-rate 0.001 --meta-step 1 --meta-step-final 0 --meta-batch 5 --meta-iters 50000 --eval-batch 5 --eval-iters 50 --checkpoint ckpt$ITER'_o55sweep' $OPTIONS 2>&1 | tee iter$ITER'_o55sweep.log'

    python sweep_omniglot.py --shots 1 --classes 20 --inner-batch 20 --inner-iters $ITER --meta-step 1 --meta-batch 5 --meta-iters 50000 --eval-batch 10 --eval-iters 50 --learning-rate 0.0005 --meta-step-final 0 --train-shots 10 --checkpoint ckpt$ITER'_o120sweep' $OPTIONS 2>&1 | tee iter$ITER'_o120sweep.log' 

    python sweep_omniglot.py --classes 20 --inner-batch 20 --inner-iters $ITER --meta-step 1 --meta-batch 5 --meta-iters 50000 --eval-batch 10 --eval-iters 50 --learning-rate 0.0005 --meta-step-final 0 --train-shots 10 --checkpoint ckpt$ITER'_o520sweep' $OPTIONS 2>&1 | tee iter$ITER'_o520sweep.log'

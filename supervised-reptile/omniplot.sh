OPTIONS='--transductive --eval-interval 1000 --eval-samples 500'

for ITER in 5 10 15 20 25 ; do

    python sweep_omniglot.py --shots 1 --inner-batch 10 --inner-iters $ITER --meta-step 1 --meta-batch 5 --meta-iters 25000 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 10 --checkpoint ckpt$ITER'_o15plot' $OPTIONS 2>&1 | tee iter$ITER'_o15plot.log'

    python sweep_omniglot.py --train-shots 10 --inner-batch 10 --inner-iters $ITER --learning-rate 0.001 --meta-step 1 --meta-step-final 0 --meta-batch 5 --meta-iters 25000 --eval-batch 5 --eval-iters 50 --checkpoint ckpt$ITER'_o55plot' $OPTIONS 2>&1 | tee iter$ITER'_o55plot.log'

    python sweep_omniglot.py --shots 1 --classes 20 --inner-batch 20 --inner-iters $ITER --meta-step 1 --meta-batch 5 --meta-iters 25000 --eval-batch 10 --eval-iters 50 --learning-rate 0.0005 --meta-step-final 0 --train-shots 10 --checkpoint ckpt$ITER'_o120plot' $OPTIONS 2>&1 | tee iter$ITER'_o120plot.log' 

    python sweep_omniglot.py --classes 20 --inner-batch 20 --inner-iters $ITER --meta-step 1 --meta-batch 5 --meta-iters 25000 --eval-batch 10 --eval-iters 50 --learning-rate 0.0005 --meta-step-final 0 --train-shots 10 --checkpoint ckpt$ITER'_o520plot' $OPTIONS 2>&1 | tee iter$ITER'_o520plot.log'

done

python -u run_omniglot.py --shots 1 --inner-batch 10 --inner-iters 25 --meta-step 1 --meta-batch 5 --meta-iters 100000 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 10 --checkpoint ckpt_omni15noreg --transductive 2>&1 | tee omni15noreg.log

python -u run_omniglot.py --shots 1 --classes 20 --inner-batch 20 --inner-iters 50 --meta-step 1 --meta-batch 5 --meta-iters 200000 --eval-batch 10 --eval-iters 50 --learning-rate 0.0005 --meta-step-final 0 --train-shots 10 --checkpoint ckpt_omni120noreg --transductive 2>&1 | tee omni120noreg.log

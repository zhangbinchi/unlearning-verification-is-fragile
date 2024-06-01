# Adversarial Unlearning Against Verification

The code is associated with *Verification of Machine Unlearning is Fragile*.

## Retraining-based Adversarial Unlearning

Run orginal training and naive retraining on MNIST (or CIFAR-10/SVHN) using MLP (or CNN/ResNet) for 5 times
```
python retrain_origin.py --data mnist --m mlp --t 5
```
For random mini-batch selection, run Adv-Sr with *M=50* on MNIST using MLP for 5 times
```
python Adv_Sr.py --data mnist --m mlp --t 5 --M 50
```
For nearest-neighbor mini-batch selection, run Adv-Sn on MNIST using MLP for 5 times.
First we should generate a file recording the nearest neighbors
```
python generate_dist_list.py --data mlp
```
Then run
```
python Adv_Sn.py --data mnist --m mlp --t 5
```

## Forging-based Adversarial Unlearning

First, to generate the proof of original training, run train_with_proof on MNIST (or CIFAR-10/SVHN) using MLP (or CNN/ResNet)
```
python train_with_proof.py --dataset mnist --model mlp
```
Then, to forge the proof of retraining, we generate a file recording the nearest neighbors similarly as before
```
python generate_dist_list.py --data mlp
```
And we run forge on MNIST using MLP
```
python forge.py --dataset mnist --model mlp --weight-decay 0
```
Finally, we run verify to compute the verification error of each iteration
```
python verify.py --dataset mnist --model mlp
```

### Forging Simulation
Considering the large storage requirement for storing the proof, we provide a simulation program that yields the same verification error but without the intermediate proof storing processes. A simplified version of all forging processes above is to run
```
python forge_sim.py --dataset mnist --model mlp --weight-decay 0
```

Please refer to Appendix B for more details on the hyperparameter setting.

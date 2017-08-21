## BinaryNet.tf
Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1.  implementation in tensorflow (https://papers.nips.cc/paper/6573-binarized-neural-networks)

This is incomplete training example for BinaryNets using Binary-Backpropagation algorithm as explained in 
"Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, 
on following datasets: Cifar10/100, MNIST.

Note that in this folder I didn’t implemented (yet...) shift-base BN , shift-base AdaMax (instead I just use the vanilla BN and Adam).
Likewise, I use deterministic binarization and I don't apply the initialization coefficients from GLorot&Bengio 2010.
Finally "sparse_softmax_cross_entropy_with_logits" loss is used instead if the SquareHingeLoss. 

The implementation is based on https://github.com/eladhoffer/convNet.tf but the main idea can be easily transferred to any tensorflow wrapper
(e.g., slim,keras)

## Data
This implementation supports cifar10/cifar100 and mnist (without preprocessing) 

## Dependencies
tensorflow version 1.2.1

## Training

* Train cifar10 model using gpu:
 python main.py --model BNN_vgg_cifar10 --save BNN_cifar10 --dataset cifar10 --gpu True
* Train cifar10 model using cpu:
 python main.py --model BNN_vgg_cifar10 --save BNN_cifar10 --dataset cifar10


## Results
Cifar10 should reach at least 88% top-1 accuracy







# Camel: Communication-Efficient and Maliciously Secure Federated Learning in the Shuffle Model of Differential Privacy

This implementation accompanies our paper by Shuangqing Xu, Yifeng Zheng and Zhongyun Hua to appear at ACM CCS'24.

## Usage

Download the repository and install all required packages as listed in requirements.txt.

#### Differentially Private Model Training on MNIST

To evaluate the model utility on MNIST, run the following command

``` shell
python train_mnist.py --epsilon 1.9 --epochs 1000 --lr 0.1 --clip-val 0.5 --batch-size 12800 --device cuda
```

#### Differentially Private Model Training on FMNIST

To evaluate the model utility on FMNIST, run the following command

``` shell
python train_fmnist.py --epsilon 2 --epochs 1000 --lr 0.1 --clip-val 0.5 --batch-size 12800 --device cuda
```

#### Maliciously Secure Secret-Shared Shuffle

To run the maliciously secure secret-shared shuffle, run the following command

``` shell
cd Secure-Shuffling
./shufflemessage
```

**WARNING**: This is an academic proof-of-concept prototype and has not received careful code review. This implementation is NOT ready for production use.

## Acknowledgments

- Part of the maliciously secure secret-shared shuffle implementation is based on the public implementation of the Clarion [paper](https://www.ndss-symposium.org/ndss-paper/auto-draft-243/) with code repo [here](https://github.com/SabaEskandarian/Clarion)
- Part of our differentially private model training implementation is extended from the public implementation in Asi et al.'s [paper](https://arxiv.org/abs/2306.04444) with code repo [here](https://github.com/apple/ml-projunit)

# Chapter 11: Training NNs

## Review

- [ ] Make notes on various optimizers and their differences, advantages and disadvantages. (handwritten + excalidraw diagrams)
  - [x] [SGD w/ momentum (Andrew Ng)](https://www.youtube.com/watch?v=k8fTYJPd3_I&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=20)
  - [x] [SebRashka L12](https://www.youtube.com/watch?v=7RhNXYqDBfU)
  - [ ] Terminal velocity [Gemini chat](https://gemini.google.com/share/6859e5761a44)
  - [ ] Weight decay and AdamW [Weight Decay is not L2](https://www.johntrimble.com/posts/weight-decay-is-not-l2-regularization/)
- [ ] Initialisation strategies: Make notes on how Glorot, He and LeCun initialisation work in terms of derivation

## Book Questions

### 1. What is the problem that Glorot initialisation and He Initialisation aim to fix?

Initialisation strategies aim at speeding up the training of NNs by ensuring that the weights are set in a way that allows for efficient learning. Glorot initialisation (also known as Xavier initialisation) and He initialisation are designed to address the problem of vanishing and exploding gradients, which can occur when training deep neural networks - this is because, if gradients have wildly varying magnitudes, it can lead to divergence as updates become too large or too small.

They do so by setting the initial weights such that the standard deviation of the input layer is equal/close to the standard deviation of the output layer, which helps maintain a stable gradient flow during training. Glorot initialisation is typically used for activation functions like sigmoid and tanh, while He initialisation is more suitable for ReLU activations.


### 3. Is it okay to initialise the bias terms to 0?

Yes - It doesn't matter as much as the weights, because the bias terms don't affect the variance of the output. For some activation functions it might be beneficial to initialise the bias terms to be non-zero in terms of speeding up convergence, but in general, it won't matter much.

### 5. What may happen if you set the momentum hyperparameter too close to 1 (e.g 0.99999) when using an SGD optimizer? 


# Playing Pong using Deep Reinforcement Learning

This repository is based on a Nature paper published in ... by Google Deepmind. To familiarize myself with deep learning, reinforcement learning I've implemented the algorhighm described in the methods section of the Nature paper from scratch using PyGame and Tensorflow.

## Learing to play Pong with Deep Q learning

In Deep Q learning, we aim to train a [convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network) (CNN) that slowly learns how to play a game. The algorighm starts of with random play, and based on the rewards (winning or losing) the network slowly learns playing moves to progress to games states with a higher probability to win.

The CNN has only raw pixels as input, it does not have any prior knowledge about the game.

!['hoi'](images/input_image_example.png)

Pong is implemented in pygame (for visuals) and numpy (for training speed).

## Installing dependencies

Install dependencies in a new anaconda environment and activate the environment:

```
conda env create -f environment.yml
conda activate pong_rf
```

## Training the network

Reinforcement and CNN hyperparameters are specified in the params.yml. If hyperparamters are provided in a list, a grid search will automatically be initiated and results are stored in subdirectories. To excecute training and store results in the exp subdirectory run:

`python pong_rf.py params.yml exp`

Follow result stats bin tensorboard by:

`tensorboard --logdir exp --host localhost --port 8008`



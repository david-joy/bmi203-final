#!/usr/bin/env python

""" Train an 8-3-8 autoencoder """

import matplotlib.pyplot as plt

import numpy as np

from final_project import model, layers

# Constants
LEARN_RATE = 0.0001
WEIGHT_DECAY = 0.0001
ACTIVATION = 'relu'
EPOCHS = 4000

# Training data
train_x = [
    np.array([1, 0, 0, 0, 0, 0, 0, 0]),
    np.array([0, 1, 0, 0, 0, 0, 0, 0]),
    np.array([0, 0, 1, 0, 0, 0, 0, 0]),
    np.array([0, 0, 0, 1, 0, 0, 0, 0]),
    np.array([0, 0, 0, 0, 1, 0, 0, 0]),
    np.array([0, 0, 0, 0, 0, 1, 0, 0]),
    np.array([0, 0, 0, 0, 0, 0, 1, 0]),
    np.array([0, 0, 0, 0, 0, 0, 0, 1]),
] * 20

# Model
autoenc = model.Model('3-8-3 Autoencoder', input_size=8)
autoenc.add_layer(layers.FullyConnected(size=3, func=ACTIVATION))
autoenc.add_layer(layers.FullyConnected(size=8, func=ACTIVATION))
autoenc.init_weights()

learn_rate = LEARN_RATE

losses = []
for epoch in range(EPOCHS):
    epoch_losses = []
    for x in train_x:
        err = autoenc.gradient_descent(x, x,
                                       learn_rate=learn_rate,
                                       weight_decay=WEIGHT_DECAY)
        epoch_losses.append(err)
    losses.append(np.mean(epoch_losses))

    if epoch % 100 == 0:
        print('Epoch: {:04d}: Last error: {}'.format(epoch, losses[-1]))

print('Saving final weights...')
autoenc.save_weights('weights/autoencoder_838_{}.npz'.format(ACTIVATION))

losses = np.array(losses)

fig, ax = plt.subplots(1, 1, figsize=(12, 12))

ax.plot(np.arange(losses.shape[0]), losses, linewidth=3)
ax.set_xlabel('Epochs')
ax.set_ylabel('Mean Squared Error')
ax.set_title('Loss for 8-3-8 Autoencoder with {}'.format(ACTIVATION))

fig.savefig('plots/autoencoder_838_{}.png'.format(ACTIVATION))
plt.close()

"""
Author: Noah Thurston
Date: 5/10/2020


Code written for the ECE523 final project. 
https://github.com/noahthurston/VAE-Password-Prediction

Code reference: https://blog.keras.io/building-autoencoders-in-keras.html

This was originally ran as a google colab notebook, so it will take some re-working
to run natively in python. 
"""



import keras
import numpy as np
from google.colab import files

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense, Reshape, Flatten, Activation, SimpleRNN
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy, categorical_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import os
import itertools
import nltk


# upload the .txt file
from google.colab import files
uploaded = files.upload()

import io

file_name = 'top1M.txt'

loaded_file = io.BytesIO(uploaded[file_name]).readlines()

loaded_file = [i.decode("utf-8").strip() for i in loaded_file]




# start cleaning data
unknown_token = '_'
passwords_raw = loaded_file

vocab_size = 90

password_size = 10
password_append = [unknown_token for i in range(password_size)]
passwords_split = [list(password) for password in passwords_raw]


# create dictionary of words
char_freq = nltk.FreqDist(itertools.chain(*passwords_split))


print("char_freq:\n{}".format(char_freq))
print("Found %d unique characters." % len(char_freq.items()))

character_vocab = char_freq.most_common(vocab_size - 1)
print("character_vocab:\n{}".format(character_vocab))

def sample(M, p):
  return [single_sample(p) for _ in range(M)]
  
def single_sample(cdf): 
  # create cdf  
  # randomly uniformly sample
  uniform_sample = np.random.uniform(0, 1)
  
  p_sample_index = 0
  while(uniform_sample>cdf[p_sample_index]):
    p_sample_index += 1
  return p_sample_index


pdf = [char_freq.freq(char) for char in [index_to_char[index] for index in range(vocab_size)]]
print(pdf)
cdf = [np.sum(pdf[:ind+1]) for ind in range(len(pdf))] 

single_sample(cdf)

index_to_char = [x[0] for x in character_vocab]
index_to_char.append(unknown_token)
char_to_index = dict([c, i] for i, c in enumerate(index_to_char))

print("index_to_char:\n{}".format(index_to_char))
print("char_to_index:\n{}".format(char_to_index))



# replace all words not in our vocabulary with the unknown token
# add start and end sentence tokens
for i, password in enumerate(passwords_split):
    passwords_split[i] = ([w if w in char_to_index else unknown_token for w in password] + password_append)[:password_size]

# print("passwords_split:\n{}".format(passwords_split))

passwords_onehot = [[keras.utils.to_categorical(char_to_index[char], num_classes=vocab_size) for char in password] for password in passwords_split]

passwords_onehot = np.array(passwords_onehot)

print(passwords_onehot.shape)


# reparameterization trick
def sampling(args):


    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon



def show_img(file_string):
  img=mpimg.imread(file_string)
  imgplot = plt.imshow(img)
  plt.show()
  # files.download(file_string)


passwords_onehot_flat = passwords_onehot.reshape((-1, password_size*vocab_size))
np.random.shuffle(passwords_onehot_flat)

num_samples = len(passwords_onehot_flat)
print("num_samples: {}".format(num_samples))

spit_index = int(num_samples*0.90)
print("spit_index: {}".format(spit_index))


x_train = passwords_onehot_flat[:spit_index]
x_test = passwords_onehot_flat[spit_index:]


print("len(x_train): {}".format(len(x_train)))
print("len(x_test): {}".format(len(x_test)))



x_train = np.reshape(x_train, [-1, password_size, vocab_size])
x_test = np.reshape(x_test, [-1, password_size, vocab_size])

input_shape = (password_size, vocab_size, )



original_dim = password_size*vocab_size
intermediate_dim = 300
batch_size = 10
latent_dim = 7
epochs = 1


# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')

# intermediate_dim
x = Dense(30, activation='relu')(inputs)

x = Flatten()(x)

z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)
show_img('vae_mlp_encoder.png')

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)

outputs = Dense(password_size*vocab_size, activation=None)(x)

outputs = Reshape((password_size, vocab_size))(outputs)
outputs = Activation('softmax')(outputs)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)
show_img('vae_mlp_decoder.png')


# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')



parser = argparse.ArgumentParser()
help_ = "Load h5 model trained weights"
parser.add_argument("-w", "--weights", help=help_)

import sys
sys.argv=['']
del sys

args = parser.parse_args()
models = (encoder, decoder)






reconstruction_loss = categorical_crossentropy(inputs, outputs)


reconstruction_loss *= original_dim
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()
plot_model(vae, to_file='vae_mlp.png', show_shapes=True)
show_img('vae_mlp.png')


loading_weights = False
if not loading_weights:
  vae.fit(x_train,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=(x_test, None))
  vae.save_weights('vae_mlp_mnist.h5')
else:
  file_name = "10epoch_7latent.h5"
  uploaded = files.upload()
  vae.load_weights(file_name)




files.download('vae_mlp_mnist.h5')

def word_to_onehot(word):
  return np.array([keras.utils.to_categorical(char_to_index[char], num_classes=vocab_size) for char in word]).reshape((1, 10, 90))

def sparse_to_onehot(sparse_vector):
  return np.array([keras.utils.to_categorical(index, num_classes=vocab_size) for index in sparse_vector]).reshape((1, 10, 90))

def onehot_to_word(onehot_vector):
  onehot_vector = onehot_vector.reshape(password_size, vocab_size)
  vector_argmaxed = onehot_vector.argmax(axis=-1)

  letters = [index_to_char[index] for index in vector_argmaxed]

  password = ''.join(letters)
  return password

def calc_error(word1, word2):
  assert len(word1) == len(word2)
  return np.sum([1 if word1[i]!=word2[i] else 0 for i in range(len(word1)) ])





print("\nTest set passwords:")
l1_errors = []
for _ in range(10): 
  custom_word_onehot = x_test[np.random.randint(len(x_test))]

  reconstructed = vae.predict(custom_word_onehot.reshape((1, password_size, vocab_size)))
  reconstructed_as_string = onehot_to_word(reconstructed)

  l1_error = calc_error(onehot_to_word(custom_word_onehot), reconstructed_as_string)
  l1_errors.append(l1_error)

  print("{} {} {}".format(onehot_to_word(custom_word_onehot), reconstructed_as_string, l1_error))
print("avg error: {}".format(np.mean(l1_errors)))





print("\nRandomly generated passwords (from char pdf):")
l1_errors = []
for _ in range(100): 
  custom_sparse_vector = [single_sample(cdf) for _ in range(password_size)]
  custom_word_onehot = sparse_to_onehot(custom_sparse_vector)


  reconstructed = vae.predict(custom_word_onehot.reshape((1, password_size, vocab_size)))
  reconstructed_as_string = onehot_to_word(reconstructed)

  l1_error = calc_error(onehot_to_word(custom_word_onehot), reconstructed_as_string)
  l1_errors.append(l1_error)

  print("{} {} {}".format(onehot_to_word(custom_word_onehot), reconstructed_as_string, l1_error))
print("avg L1 error: {}".format(np.mean(l1_errors)))





print("\nRandomly generated passwords (uniform):")
l1_errors = []
for _ in range(100): 
  custom_sparse_vector = [single_sample(cdf) for _ in range(password_size)]

  custom_sparse_vector = np.random.randint(vocab_size, size=password_size)
  custom_word_onehot = sparse_to_onehot(custom_sparse_vector)


  reconstructed = vae.predict(custom_word_onehot.reshape((1, password_size, vocab_size)))
  reconstructed_as_string = onehot_to_word(reconstructed)

  l1_error = calc_error(onehot_to_word(custom_word_onehot), reconstructed_as_string)
  l1_errors.append(l1_error)

  print("{} {} {}".format(onehot_to_word(custom_word_onehot), reconstructed_as_string, l1_error))
print("avg L1 error: {}".format(np.mean(l1_errors)))








# vector isolation
# trying to isolate the 'l33t speak'vector that swaps 'o's for '0's
z_sample_upper = encoder.predict(word_to_onehot("s00n______").reshape(1, password_size, vocab_size))[2]
reconstructed_onehot_upper = decoder.predict(z_sample_upper.reshape((1, 7)))

z_sample_lower = encoder.predict(word_to_onehot("soon______").reshape(1, password_size, vocab_size))[2]
reconstructed_onehot_lower = decoder.predict(z_sample_lower.reshape((1, 7)))

cap_vector = z_sample_upper - z_sample_lower



# vector isolation example
# trying to isolate the vector at append '123'
z_sample_upper = encoder.predict(word_to_onehot("darn123___").reshape(1, password_size, vocab_size))[2]
reconstructed_onehot_upper = decoder.predict(z_sample_upper.reshape((1, 7)))

z_sample_lower = encoder.predict(word_to_onehot("darn______").reshape(1, password_size, vocab_size))[2]
reconstructed_onehot_lower = decoder.predict(z_sample_lower.reshape((1, 7)))

count_vector = z_sample_upper - z_sample_lower


z_sample_b = encoder.predict(word_to_onehot("loot______").reshape(1, password_size, vocab_size))[2]
z_sample_B = z_sample_b + cap_vector + count_vector

reconstructed_onehot_B = decoder.predict(z_sample_B.reshape((1, 7)))

reconstructed_password_B = onehot_to_word(reconstructed_onehot_B)

print(reconstructed_password_B) # should be "loot123___"



# interpolation example
# interpolating between two vectors in the latent space
starting_word = "noah123___"
ending_word = "noaht1990_"

starting_latent = encoder.predict(word_to_onehot(starting_word))[2]
ending_latent = encoder.predict(word_to_onehot(ending_word))[2]

latent_difference = ending_latent - starting_latent


num_steps = 15
step_vector = latent_difference / num_steps


for step_index in range(num_steps+1):
  current_latent_vector = starting_latent + step_vector * step_index
  current_onehot = decoder.predict(current_latent_vector)
  current_word = onehot_to_word(current_onehot)

  print("{}:\t{}".format(step_index, current_word))





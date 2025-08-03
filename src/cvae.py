#!/usr/bin/env python
# coding: utf-8

# ##### Copyright 2020 The TensorFlow Authors.

# In[ ]:


# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# # Convolutional Variational Autoencoder

# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/tutorials/generative/cvae">
#     <img src="https://www.tensorflow.org/images/tf_logo_32px.png" />
#     View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/cvae.ipynb">
#     <img src="https://www.tensorflow.org/images/colab_logo_32px.png" />
#     Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/cvae.ipynb">
#     <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
#     View source on GitHub</a>
#   </td>
#   <td>
#     <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/tutorials/generative/cvae.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
#   </td>
# </table>

# This notebook demonstrates how to train a Variational Autoencoder (VAE) ([1](https://arxiv.org/abs/1312.6114), [2](https://arxiv.org/abs/1401.4082)) on the MNIST dataset. A VAE is a probabilistic take on the autoencoder, a model which takes high dimensional input data and compresses it into a smaller representation. Unlike a traditional autoencoder, which maps the input onto a latent vector, a VAE maps the input data into the parameters of a probability distribution, such as the mean and variance of a Gaussian. This approach produces a continuous, structured latent space, which is useful for image generation.
#
# ![CVAE image latent space](images/cvae_latent_space.jpg)

# ## Setup

# In[ ]:


get_ipython().system("pip install tensorflow-probability")

# to generate gifs
get_ipython().system("pip install imageio")
get_ipython().system("pip install git+https://github.com/tensorflow/docs")


# In[ ]:


from IPython import display

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_probability as tfp
import time
import sys
import os  # Added import for the os module


# ## Load the MNIST dataset
# Each MNIST image is originally a vector of 784 integers, each of which is between 0-255 and represents the intensity of a pixel. Model each pixel with a Bernoulli distribution in our model, and statically binarize the dataset.

# In[ ]:


(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()


# In[ ]:


def preprocess_images(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.0
    return np.where(images > 0.5, 1.0, 0.0).astype("float32")


train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)


# In[ ]:


train_size = 60000
batch_size = 32
test_size = 10000


# ## Use *tf.data* to batch and shuffle the data

# In[ ]:


train_dataset = (
    tf.data.Dataset.from_tensor_slices(train_images)
    .shuffle(train_size)
    .batch(batch_size)
)
test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_images).shuffle(test_size).batch(batch_size)
)


# ## Define the Encoder and Decoder Networks
# The CVAE model is refactored into two separate classes for better modularity and clarity.
#
# ### VAE Encoder
# The encoder network, `VaeEncoder`, takes input data (e.g., an image) and maps it to the parameters of a probability distribution in the latent space. In this implementation, it models the latent distribution as a diagonal Gaussian and outputs the mean vector ($\mu$) and the log-variance vector ($\log \sigma^2$).
#
# ### VAE Decoder
# The decoder network, `VaeDecoder`, takes a sample from the latent space ($z$) and reconstructs the original data. It defines the conditional distribution of the observation $p(x|z)$.

# In[ ]:


class VaeEncoder(tf.keras.Model):
    """
    The Encoder network for the Variational Autoencoder.

    This network takes an input image and processes it through a series of
    convolutional layers to produce the parameters for a latent space distribution,
    specifically the mean (mu) and log-variance (logvar) of a Gaussian distribution.

    The architecture consists of two Conv2D layers with 'relu' activation,
    followed by a Flatten layer and a final Dense layer. The output of the Dense
    layer is split to provide the mu and logvar vectors for the latent space.
    """

    def __init__(self, latent_dim):
        super(VaeEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation="relu"
                ),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation="relu"
                ),
                tf.keras.layers.Flatten(),
                # No activation on the final dense layer to directly output
                # the raw values for mean and log-variance
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

    def call(self, x):
        """Passes input through the encoder network and returns mean and logvar."""
        mean, logvar = tf.split(self.encoder_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar


class VaeDecoder(tf.keras.Model):
    """
    The Decoder network for the Variational Autoencoder.

    This network takes a sampled latent vector (z) and reconstructs an image
    by passing it through a series of dense and convolutional transpose layers.

    The architecture is a mirror of the encoder, starting with a Dense layer
    followed by a Reshape and then two Conv2DTranspose layers. The final
    Conv2DTranspose layer with a single filter outputs the reconstructed image logits.
    """

    def __init__(self, latent_dim):
        super(VaeDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.decoder_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation="relu",
                ),
                # No activation on the final layer to output raw logits
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=1, padding="same"
                ),
            ]
        )

    def call(self, z, apply_sigmoid=False):
        """Passes a latent vector through the decoder and returns logits or probabilities."""
        logits = self.decoder_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


class CVAE(tf.keras.Model):
    """
    Convolutional Variational Autoencoder (CVAE) model.

    This class orchestrates the encoder and decoder networks to implement
    the full VAE functionality, including the reparameterization trick and
    methods for sampling and generating images.
    """

    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        # Instantiate the separate encoder and decoder networks
        self.encoder = VaeEncoder(latent_dim)
        self.decoder = VaeDecoder(latent_dim)

    @tf.function
    def sample(self, eps=None):
        """
        Generates new images by sampling from the latent space prior.

        This method is used for inference/generation after the model is trained.
        If no latent vector `eps` is provided, it samples from a standard normal
        distribution, which is the VAE's prior distribution.
        """
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        """
        Encodes an input image `x` into the parameters of the latent distribution.

        This is the forward pass through the VaeEncoder network.
        """
        mean, logvar = self.encoder(x)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        """
        Performs the reparameterization trick to sample a latent vector `z`.

        This is a crucial step that allows for backpropagation through the
        stochastic sampling process. It computes z from the mean and log-variance
        of the latent distribution and a random noise epsilon.

        $z = \mu + \sigma \cdot \epsilon$
        """
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z, apply_sigmoid=False):
        """
        Decodes a latent vector `z` to reconstruct an image.

        This is the forward pass through the VaeDecoder network.
        """
        return self.decoder(z, apply_sigmoid)


# ## Define the loss function and the optimizer
#
# VAEs train by maximizing the evidence lower bound (ELBO) on the marginal log-likelihood:
#
# $$\log p(x) \ge \text{ELBO} = \mathbb{E}_{q(z|x)}\left[\log \frac{p(x, z)}{q(z|x)}\right].$$
#
# In practice, optimize the single sample Monte Carlo estimate of this expectation:
#
# $$\log p(x| z) + \log p(z) - \log q(z|x),$$
# where $z$ is sampled from $q(z|x)$.
#
# The ELBO consists of two key terms:
# 1. **Reconstruction Term ($\log p(x|z)$):** This term measures how well the decoder can reconstruct the input data from its latent representation. We use a binary cross-entropy loss for this purpose.
# 2. **KL Divergence Term ($\log p(z) - \log q(z|x)$):** This term acts as a regularizer, forcing the encoder's learned latent distribution ($q(z|x)$) to be close to a simple prior distribution ($p(z)$), which is a standard normal distribution. This ensures a continuous and well-structured latent space, enabling effective generation.

# In[ ]:


optimizer = tf.keras.optimizers.Adam(1e-4)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis
    )


def compute_loss(model, x):
    """
    Computes the negative ELBO loss for a single batch of data.

    The loss is composed of a reconstruction term and a KL divergence term.
    """
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    # Reconstruction loss (Binary Cross-Entropy)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])

    # KL Divergence loss (distance between latent distribution and prior)
    logpz = log_normal_pdf(z, 0.0, 0.0)
    logqz_x = log_normal_pdf(z, mean, logvar)

    # The total loss is the negative of the ELBO
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer):
    """
    Executes one training step and applies gradients to the model.

    This function computes the loss and gradients using a GradientTape, and
    then uses the optimizer to update the model's trainable variables.
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


# ## Training
#
# * Start by iterating over the dataset
# * During each iteration, pass the image to the encoder to obtain a set of mean and log-variance parameters of the approximate posterior $q(z|x)$
# * then apply the *reparameterization trick* to sample from $q(z|x)$
# * Finally, pass the reparameterized samples to the decoder to obtain the logits of the generative distribution $p(x|z)$
# * Note: Since you use the dataset loaded by keras with 60k datapoints in the training set and 10k datapoints in the test set, our resulting ELBO on the test set is slightly higher than reported results in the literature which uses dynamic binarization of Larochelle's MNIST.
#
# ### Generating images
#
# * After training, it is time to generate some images
# * Start by sampling a set of latent vectors from the unit Gaussian prior distribution $p(z)$
# * The generator will then convert the latent sample $z$ to logits of the observation, giving a distribution $p(x|z)$
# * Here, plot the probabilities of Bernoulli distributions
#

# In[ ]:


epochs = 10
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 2
num_examples_to_generate = 16

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim]
)
model = CVAE(latent_dim)


# In[ ]:


def generate_and_save_images(model, epoch, test_sample):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap="gray")
        plt.axis("off")

    # tight_layout minimizes the overlap between 2 sub-plots
    filename = "image_at_epoch_{:04d}.png".format(epoch)
    plt.savefig(filename)
    print(f"Saved generated image to: {os.path.abspath(filename)}")
    # The plt.show() command is commented out because it opens a GUI window,
    # which may not be desirable or possible in a containerized environment.
    # If you want to see the images, you can uncomment this line.
    # plt.show()


# In[ ]:


# Pick a sample of the test set for generating output images
assert batch_size >= num_examples_to_generate
for test_batch in test_dataset.take(1):
    test_sample = test_batch[0:num_examples_to_generate, :, :, :]


# In[ ]:


generate_and_save_images(model, 0, test_sample)

for epoch in range(1, epochs + 1):
    start_time = time.time()
    for train_x in train_dataset:
        train_step(model, train_x, optimizer)
    end_time = time.time()

    loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
        loss(compute_loss(model, test_x))
    elbo = -loss.result()
    display.clear_output(wait=False)
    print(
        "Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}".format(
            epoch, elbo, end_time - start_time
        )
    )
    # Ensure the print statement is flushed immediately
    sys.stdout.flush()
    generate_and_save_images(model, epoch, test_sample)


# ### Display a generated image from the last training epoch
#
# This section is commented out to avoid using GUI elements which may not be available in a container.

# In[ ]:


# def display_image(epoch_no):
#   return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


# In[ ]:


# plt.imshow(display_image(epoch))
# plt.axis('off')  # Display images


# ### Display an animated GIF of all the saved images
#
# This section is also commented out to avoid using GUI elements.

# In[ ]:


anim_file = "cvae.gif"

with imageio.get_writer(anim_file, mode="I") as writer:
    filenames = glob.glob("image*.png")
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

print(f"Saved animation to: {os.path.abspath(anim_file)}")


# In[ ]:


# This is commented out because embed.embed_file requires a front-end environment
# which is not present in a headless container.
# import tensorflow_docs.vis.embed as embed
# embed.embed_file(anim_file)


# ### Display a 2D manifold of digits from the latent space
#
# Running the code below will show a continuous distribution of the different digit classes, with each digit morphing into another across the 2D latent space. Use [TensorFlow Probability](https://www.tensorflow.org/probability) to generate a standard normal distribution for the latent space.

# In[ ]:


def plot_latent_images(model, n, digit_size=28):
    """Plots n x n digit images decoded from the latent space."""

    norm = tfp.distributions.Normal(0, 1)
    grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
    grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
    image_width = digit_size * n
    image_height = image_width
    image = np.zeros((image_height, image_width))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z = np.array([[xi, yi]])
            x_decoded = model.sample(z)
            digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
            image[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit.numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap="Greys_r")
    plt.axis("Off")
    # The plt.show() command is commented out for container compatibility.
    # plt.show()
    # Instead, save the plot to a file if you need to access it.
    filename = "latent_space_plot.png"
    plt.savefig(filename)
    print(f"Saved latent space plot to: {os.path.abspath(filename)}")
    plt.close()  # Close the figure to free up memory


# In[ ]:


plot_latent_images(model, 20)


# ## Next steps
#
# This tutorial has demonstrated how to implement a convolutional variational autoencoder using TensorFlow.
#
# As a next step, you could try to improve the model output by increasing the network size.
# For instance, you could try setting the `filter` parameters for each of the `Conv2D` and `Conv2DTranspose` layers to 512.
# Note that in order to generate the final 2D latent image plot, you would need to keep `latent_dim` to 2. Also, the training time would increase as the network size increases.
#
# You could also try implementing a VAE using a different dataset, such as CIFAR-10.
#
# VAEs can be implemented in several different styles and of varying complexity. You can find additional implementations in the following sources:
# - [Variational AutoEncoder (keras.io)](https://keras.io/examples/generative/vae/)
# - [VAE example from "Writing custom layers and models" guide (tensorflow.org)](https://www.tensorflow.org/guide/keras/custom_layers_and_models#putting_it_all_together_an-end-to-end-example)
# - [TFP Probabilistic Layers: Variational Auto Encoder](https://www.tensorflow.org/probability/examples/Probabilistic_Layers_VAE)
#
# If you'd like to learn more about the details of VAEs, please refer to [An Introduction to Variational Autoencoders](https://arxiv.org/abs/1906.02691).

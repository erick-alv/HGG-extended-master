import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()
@tf.function
def reparametrization_trick(args):#important we are using this in the Layer Lambsa; therefore the inputs this functions receives are tensors
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """
    mu, log_sigma = args
    epsilon = tf.random_normal(shape=tf.shape(log_sigma), mean=0, stddev=1, dtype=tf.float32)
    z = mu + tf.exp(0.5 * log_sigma) * epsilon
    return z

@tf.function
def loss_function(batch, img, mu, log_sigma, beta, imsize, input_channels):  # important this function receives actual tensors
    # reconstruction loss
    n = tf.cast(tf.shape(mu)[0], dtype=tf.float32)
    flat_input = tf.reshape(batch, [-1, imsize * imsize * input_channels])
    flat_output = tf.reshape(img, [-1, imsize * imsize * input_channels])
    rec_loss = tf.reduce_sum(tf.square(flat_output - flat_input)) / n

    # kl_loss
    kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(
        1 + log_sigma - tf.square(mu) - tf.exp(log_sigma), axis=1))
    loss = rec_loss + beta * kl_loss
    return rec_loss, kl_loss, loss


class CVAE_Keras:
    def __init__(self, lr, input_channels, representation_size=16, imsize=84, num_filters=8, beta=2.5, restore_path=None):
        self.representation_size = representation_size
        self.input_channels = input_channels
        self.imsize = imsize
        self.imlength = self.imsize ** 2 * self.input_channels
        self.lr = lr
        self.restore_path = restore_path
        self.beta = beta
        self.num_filters = num_filters
        self.create_model()


    def create_model(self):

        def encoder_functional_net(X):
            x = tf.keras.layers.ZeroPadding2D(padding=(1, 1), data_format='channels_last')(X)
            x = tf.keras.layers.Conv2D(filters=self.num_filters, kernel_size=4, strides=2,
                                       padding='valid', use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)#todo perhaps must use trainig true or not
            x = tf.keras.layers.Activation(activation=tf.nn.leaky_relu)(x)
            x = tf.keras.layers.Conv2D(filters=self.num_filters*2, kernel_size=4, strides=2,
                                       padding='valid', use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)  # todo perhaps must use trainig true or not
            x = tf.keras.layers.Activation(activation=tf.nn.leaky_relu)(x)
            x = tf.keras.layers.ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)
            x = tf.keras.layers.Conv2D(filters=self.num_filters * 4, kernel_size=4, strides=2,
                                       padding='valid', use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)  # todo perhaps must use trainig true or not
            x = tf.keras.layers.Activation(activation=tf.nn.leaky_relu)(x)
            x = tf.keras.layers.ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)
            x = tf.keras.layers.Conv2D(filters=self.num_filters * 8, kernel_size=4, strides=2,
                                       padding='valid', use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)  # todo perhaps must use trainig true or not
            x = tf.keras.layers.Activation(activation=tf.nn.leaky_relu)(x)
            x = tf.keras.layers.Flatten()(x)
            mu = tf.keras.layers.Dense(units=self.representation_size)(x)
            log_sigma = tf.keras.layers.Dense(units=self.representation_size)(x)
            # Reparametrization trick
            z = tf.keras.layers.Lambda(function=reparametrization_trick)([mu, log_sigma])
            return z, mu, log_sigma

        def decoder_functional_net(z):
            x = tf.keras.layers.Dense(units=self.num_filters*8*(self.imsize//16)*(self.imsize//16),
                                      activation=tf.nn.relu)(z)
            x = tf.keras.layers.Reshape(target_shape=(self.imsize//16, self.imsize//16, self.num_filters*8))(x)
            x = tf.keras.layers.Conv2DTranspose(filters=self.num_filters*4, kernel_size=4, strides=1,
                                                padding='valid', use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(activation=tf.nn.relu)(x)
            x = tf.keras.layers.Conv2DTranspose(filters=self.num_filters * 2, kernel_size=4, strides=2,
                                                padding='valid', use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(activation=tf.nn.relu)(x)
            x = tf.keras.layers.ZeroPadding2D(padding=(1,1))(x)
            x = tf.keras.layers.Conv2DTranspose(filters=self.num_filters, kernel_size=3, strides=2,
                                                padding='valid', use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(activation=tf.nn.relu)(x)
            img = tf.keras.layers.Conv2DTranspose(filters=self.input_channels, kernel_size=4, strides=2,
                                                padding='valid', use_bias=False)(x)
            return img


        if tf.test.gpu_device_name():
            print('using gpu')
            device_name = "/gpu:0"
        else:
            device_name = "/cpu:0"
        #with tf.Graph().as_default():
        with tf.device(device_name):
            # creates inputs
            images_input = tf.keras.layers.Input(shape=(self.imsize, self.imsize, self.input_channels,))
            latents_input = tf.keras.layers.Input(shape=(self.representation_size,))
            # creates outputs
            z, mu, log_sigma = encoder_functional_net(images_input)
            img = decoder_functional_net(latents_input)
            # creates the models
            self.encoder = tf.keras.models.Model(images_input, [z, mu, log_sigma])
            self.decoder = tf.keras.models.Model(latents_input, img)
            self.whole_net = tf.keras.models.Model(images_input,
                                                   self.decoder(self.encoder(images_input)[0]))
            self.loss_fn = loss_function
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)#create_saver()
        if self.restore_path:
            self.load_weights(self.restore_path)

    def train(self, batch):
        with tf.GradientTape() as tape:
            z, mu, log_sigma = self.encoder(batch, training=True)
            img = self.decoder(z, training=True)
            rec_loss, kl_loss, loss = self.loss_fn(batch, img, mu, log_sigma,
                                                   self.beta, self.imsize, self.input_channels)
        grads = tape.gradient(loss, self.whole_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.whole_net.trainable_variables))
        return rec_loss.numpy(), kl_loss.numpy(), loss.numpy()

    def evaluate(self, batch):
        z, mu, log_sigma = self.encoder(batch, training=False)
        img = self.decoder(z, training=False)
        rec_loss, kl_loss, loss = self.loss_fn(batch, img, mu, log_sigma,
                                               self.beta, self.imsize, self.input_channels)
        return rec_loss.numpy(), kl_loss.numpy(), loss.numpy()

    def encode(self, im):
        im = np.array([im])
        z, mu, log_sigma = self.encoder(im, training=False)
        return z.numpy()[0], mu.numpy()[0], log_sigma.numpy()[0]
    def encode_batch(self, im):
        z, mu, log_sigma = self.encoder(im, training=False)
        return z.numpy(), mu.numpy(), log_sigma.numpy()

    def decode(self, z):
        z = np.array([z])
        img = self.decoder(z, training=False)
        return img.numpy()[0]

    def decode_batch(self, z):
        img = self.decoder(z, training=False)
        return img.numpy()

    def save_weights(self, checkpoint_filepath):
        self.whole_net.save_weights(checkpoint_filepath)

    def load_weights(self, checkpoint_filepath):
        self.whole_net.load_weights(checkpoint_filepath)
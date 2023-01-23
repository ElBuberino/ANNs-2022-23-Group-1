# Import packages
import tensorflow as tf
import os
import numpy as np
import datetime
import tqdm
import urllib
categories = [line.rstrip(b'\n') for line in urllib.request.urlopen('https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt')]
category = 'candle'

# Import data
if not os.path.isdir('npy_files'):
    os.mkdir('npy_files')

url = f'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{category}.npy'
urllib.request.urlretrieve(url, f'npy_files/{category}.npy')

images = np.load(f'npy_files/{category}.npy')
print(f'{len(images)} images to train on')



# Preprocess data
def preprocess(data):
    ## images
    # generate tf.data.Dataset
    data = tf.data.Dataset.from_tensor_slices(data)
    # reshape images
    data = data.map(lambda x: (tf.reshape(x, (28,28,1))))
    # cast pixel data to float values
    data = data.map(lambda x: (tf.cast(x, tf.dtypes.float32)))
    # normalize images
    data = data.map(lambda x: ((x/128.)-1.))

    # cache, shuffle, batch, prefetch
    data = data.cache()
    data = data.shuffle(2000)
    data = data.batch(32)
    data = data.prefetch(tf.data.AUTOTUNE)

    return data

# create models
class GAN_Generator(tf.keras.Model):
    def __init__(self, dropout_rate):
        super(GAN_Generator, self).__init__()
        self.reshape = tf.keras.layers.Reshape((7, 7, 8))
        # Img size: 7x7x8
        self.convTranspose1 = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=3, strides=(2, 2), padding='same',
                                                              activation='relu',
                                                              kernel_regularizer=tf.keras.regularizers.L2)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)
        # Img size: 14x14x4
        self.convTranspose2 = tf.keras.layers.Conv2DTranspose(filters=4, kernel_size=3, strides=(2, 2), padding='same',
                                                              activation='relu',
                                                              kernel_regularizer=tf.keras.regularizers.L2)
        self.dropout4 = tf.keras.layers.Dropout(dropout_rate)
        self.batch_norm2 = tf.keras.layers.BatchNormalization(axis=-1)
        # Img size: 28x28x4
        self.out = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=(1, 1), padding='same')
        # Img size: 28x28x1

    def call(self, x, training=False):
        x = self.reshape(x)
        # print(x.shape)
        x = self.convTranspose1(x, training=training)
        x = self.dropout3(x, training=training)
        # print(x.shape)
        x = self.convTranspose2(x, training=training)
        x = self.dropout4(x, training=training)
        x = self.batch_norm2(x, training=training)
        # print(x.shape)
        x = self.out(x)
        # print(x.shape)
        return x

class Discriminator(tf.keras.Model):
    def __init__(self, dropout_rate):
        super(Discriminator, self).__init__()
        # Img size: 28x28x1
        self.conv1 = tf.keras.layers.Conv2D(filters=4, kernel_size=3, strides=(2, 2), padding='same',
                                            activation='relu', kernel_regularizer=tf.keras.regularizers.L2)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        # Img size: 14x14x4
        self.conv2 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=(2, 2), padding='same',
                                            activation='relu', kernel_regularizer=tf.keras.regularizers.L2)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.batch_norm2 = tf.keras.layers.BatchNormalization(axis=-1)
        # Img size: 7x7x8
        self.flatten = tf.keras.layers.Flatten()
        # Img size: 1x1x392
        self.batch_norm1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.out = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.L2, activation='sigmoid')
        # Img size: 10

    def call(self, x, training=False):
        #print(x.shape, type(x), x)
        x = self.conv1(x, training=training)
        x = self.dropout1(x, training=training)
        #print(x.shape)
        x = self.conv2(x, training=training)
        x = self.dropout2(x, training=training)
        #print(x.shape)
        x = self.flatten(x)
        #print(x.shape)
        x = self.batch_norm1(x, training=training)
        x = self.out(x)
        #print(x.shape)
        return x


# set metrics
@property
def metrics(self):
    return self.metrics_list

def reset_metrics(self):
    for metric in self.metrics:
        metric.reset_states()

# train step
#@tf.function
def train_step(generator, discriminator, real_img, optimizer_gen, optimizer_dis, noise_dim, gen_metric, dis_metric):
        # Training
        noise = tf.random.normal((32,noise_dim))
        loss_function_generator = tf.keras.losses.BinaryCrossentropy()
        loss_function_discriminator = tf.keras.losses.BinaryCrossentropy()

        with tf.GradientTape(persistent=True) as tape:
            # generate images
            generated_img = generator(noise, training=True)
            print(generated_img.shape, generated_img)
            # compile predictions
            prediction_fake = discriminator(generated_img, training=True)
            prediction_real = discriminator(real_img, training=True)

            # compile losses
            loss_gen = loss_function_generator(tf.ones_like(prediction_fake), prediction_fake)
            loss_dis_real = loss_function_discriminator(tf.ones_like(prediction_real), prediction_real)
            loss_dis_fake = loss_function_discriminator(tf.zeros_like(prediction_fake), prediction_fake)
            loss_dis_total = loss_dis_real + loss_dis_fake

        # compile gradients
        gradients_gen = tape.gradient(loss_gen, generator.trainable_variables)
        gradients_dis = tape.gradient(loss_dis_total, discriminator.trainable_variables)


        del tape

        # compile trainable variables
        optimizer_gen.apply_gradients(zip(gradients_gen, generator.trainable_variables))
        optimizer_dis.apply_gradients(zip(gradients_dis, discriminator.trainable_variables))

        # update loss metric
        gen_metric.update_state(loss_gen)
        dis_metric.update_state(loss_dis_total)

        return {m.name: m.result() for m in gen_metric},  {m.name: m.result() for m in dis_metric}

def training_loop(generator, discriminator, real_data, epochs, optimizer_gen, optimizer_dis, noise_dim, gen_summary_writer, dis_summary_writer):
    for epoch in range(epochs):
        print(f"Epoch {epoch}:")
        for real_img in real_data:
        # Training
            my_train_step = train_step(generator=generator, discriminator=discriminator, real_img=real_img, optimizer_gen=optimizer_gen, optimizer_dis=optimizer_dis, noise_dim=noise_dim, gen_metric=tf.keras.metrics.Mean(name="loss"),  dis_metric=tf.keras.metrics.Mean(name="loss"))
            generator_metric = my_train_step[0]
            discriminator_metric = my_train_step[1]

            with gen_summary_writer.as_default():
                for metric in generator_metric:
                    tf.summary.scalar(f"{metric.name}", metric.result(), step=epoch)

            with dis_summary_writer.as_default():
                for metric in discriminator_metric:
                    tf.summary.scalar(f"{metric.name}", metric.result(), step=epoch)

        # print the metrics
        print([f"train_{key}: {value.numpy()}" for (key, value) in generator_metric.items()])
        print([f"train_{key}: {value.numpy()}" for (key, value) in discriminator_metric.items()])

        # reset all metrics
        generator.reset_metrics()
        discriminator.reset_metics()


################## MAIN ####################

# LOGS
path = 'C:/Users/accou/OneDrive/Desktop/Arbeit/Studium/Master/Semester03/IANNwTF/Week08/logs'
config_name = "GAN"
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
gen_log_path = f"{path}/{config_name}/{current_time}/gen"
dis_log_path = f"{path}/{config_name}/{current_time}/dis"
gen_summary_writer = tf.summary.create_file_writer(gen_log_path)
dis_summary_writer = tf.summary.create_file_writer(dis_log_path)

# HYPERPARAMETERS
EPOCHS = 2
NOISE_DIM = 392
num_examples_to_generate = 16
DROPOUT = 0.2
train_images = images[:10000]

#set random seed
seed = tf.random.normal([num_examples_to_generate, NOISE_DIM])

training_loop(generator=GAN_Generator(dropout_rate=DROPOUT),
              discriminator=Discriminator(dropout_rate=DROPOUT),
              real_data=preprocess(train_images),
              epochs=EPOCHS,
              optimizer_gen=tf.keras.optimizers.Adam(),
              optimizer_dis=tf.keras.optimizers.Adam(),
              noise_dim=NOISE_DIM,
              gen_summary_writer=gen_summary_writer,
              dis_summary_writer=dis_summary_writer)





from collections import deque
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Sequential, Model, load_model
from keras.callbacks import TensorBoard
from keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Conv2D, AveragePooling2D, Activation, \
    Flatten, concatenate

import tensorflow as tf
import keras.backend.tensorflow_backend as backend
import carla_config as settings
import time

import random
import numpy as np

class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

class DQNAgent:
    def __init__(self):

        self.model = Sequential()
        if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or \
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1] or \
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7] or \
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8] or \
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[9]:

            input_layer = Input(shape=[settings.state_dim, ])
            h0 = Dense(300, activation="tanh")(input_layer)
            # h1 = Dense(600, activation="linear")(h0)
            # h2 = Dense(800, activation="linear")(h1)

            output_layer = Dense(settings.N_actions, activation="linear")(h0)
            self.model = Model(input=input_layer, output=output_layer)
            self.model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
            print(self.model.summary())


            # self.model.add(Dense(300, input_shape=[21], activation="relu"))
            # self.model.add(Dense(600, activation="relu"))
            # inputs = self.model.input
            # x = self.model.output
            # x = Dense(256, activation='relu')(x)
            # predictions = Dense(3, activation='linear')(x)
            # self.model = Model(input=inputs, output=predictions)
            # self.model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])

        else:
            if (settings.CNN_MODEL == 1):  # 4_CNN

                self.model.add(Conv2D(64, (11, 11), input_shape=(settings.IM_HEIGHT_CNN, settings.IM_WIDTH_CNN, settings.IM_LAYERS), padding='same'))
                self.model.add(Activation('relu'))
                self.model.add(AveragePooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))

                self.model.add(Conv2D(64, (9, 9), padding='same'))
                self.model.add(Activation('relu'))
                self.model.add(AveragePooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))

                self.model.add(Conv2D(64, (7, 7), padding='same'))
                self.model.add(Activation('relu'))
                self.model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

                self.model.add(Conv2D(64, (5, 5), padding='same'))
                self.model.add(Activation('relu'))
                # self.model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

                self.model.add(Conv2D(64, (3, 3), padding='same'))
                self.model.add(Activation('relu'))
                # self.model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

                self.model.add(Flatten())

                inputs = self.model.input
                x = self.model.output
                x = Dense(256, activation='relu')(x)
                predictions = Dense(settings.N_actions, activation='linear')(x)
                self.model = Model(inputs=inputs, outputs=predictions)
                self.model.compile(loss="mse", optimizer=Adam(lr=0.001, decay=settings.EPSILON_DECAY), metrics=['accuracy'])

            elif (settings.CNN_MODEL == 2):  # 64x3 CNN

                self.model.add(Conv2D(64, (3, 3), input_shape=(settings.IM_HEIGHT_CNN, settings.IM_WIDTH_CNN, settings.IM_LAYERS), padding='same'))
                self.model.add(Activation('relu'))
                self.model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

                self.model.add(Conv2D(64, (3, 3), padding='same'))
                self.model.add(Activation('relu'))
                self.model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

                self.model.add(Conv2D(64, (3, 3), padding='same'))
                self.model.add(Activation('relu'))
                self.model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

                self.model.add(Flatten())

                inputs = self.model.input
                x = self.model.output
                x = Dense(256, activation='relu')(x)
                predictions = Dense(settings.N_actions, activation='linear')(x)
                self.model = Model(inputs=inputs, outputs=predictions)
                self.model.compile(loss="mse", optimizer=Adam(lr=0.001, decay=settings.EPSILON_DECAY), metrics=['accuracy'])

            elif (settings.CNN_MODEL == 3):


                self.model.add(Conv2D(16, (8, 8), input_shape=(settings.IM_HEIGHT_CNN, settings.IM_WIDTH_CNN, settings.IM_LAYERS), padding='same'))
                self.model.add(Activation('relu'))
                self.model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

                self.model.add(Conv2D(32, (4, 4), padding='same'))
                self.model.add(Activation('relu'))
                self.model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

                self.model.add(Flatten())


                inputs = self.model.input
                x = self.model.output
                x = Dense(256, activation='linear')(x)
                predictions = Dense(settings.N_actions, activation='linear')(x)
                self.model = Model(inputs=inputs, outputs=predictions)

                self.model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])

            elif (settings.CNN_MODEL == 4):

                im_input = Input(shape=(settings.IM_WIDTH_CNN, settings.IM_HEIGHT_CNN, settings.IM_LAYERS), name='in_image')

                conv0 = Conv2D(16, (7, 7), padding='same', activation='relu', name='conv0')(im_input)
                av0 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='av0')(conv0)

                conv1 = Conv2D(16, (5, 5), padding='same', activation='relu', name='conv1')(av0)
                av1 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='av1')(conv1)

                conv2 = Conv2D(16, (3, 3), padding='same', activation='relu', name='conv2')(av1)
                av2 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='av2')(conv2)

                flat = Flatten(name='flat')(av2)
                dense_flat = Dense(256, activation='relu', name='dense_flat')(flat)
                dense_flat1 = Dense(10, activation='relu', name='dense_flat1')(dense_flat)


                data_input = Input(shape=[2], name='in_data')
                dense0 = Dense(2, activation='linear')(data_input)

                merged = concatenate([dense_flat1, dense0], name='merged')
                dense1 = Dense(32, activation='relu', name='dense1')(merged)

                output_layer = Dense(settings.N_actions, activation='linear', name='output')(dense1)
                self.model = Model(input=[im_input, data_input], output=output_layer)
                self.model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])

        try:
            self.model = load_model(settings.MODEL_PATH)
            print('Modelo cargado:', settings.MODEL_PATH)
        except:
            print('Entrenamiento nuevo')

        self.target_model = self.model
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(
            maxlen=settings.REPLAY_MEMORY_SIZE)  # memory of previous actions, keep random set actions to help with volatility

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/logs_{settings.WORKING_MODE}/{settings.TRAIN_MODE}-{int(time.time())}")
        self.target_update_counter = 0  # updates after every episode
        # self.graph = tf.get_default_graph() #use for use different threads (train and predict)
        self.graph = tf.compat.v1.get_default_graph()  # use for use different threads (train and predict)

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False
        self.model.summary()
        tf.keras.utils.plot_model(self.model,
                                  to_file='NETWORKS/' + str(settings.TRAIN_MODE) + '_model.png',
                                  show_shapes=True,
                                  show_layer_names=True, rankdir='TB')

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < settings.MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, settings.MINIBATCH_SIZE)


        current_states = np.array([transition[0] for transition in minibatch])

        with self.graph.as_default():
            current_qs_list = self.model.predict(current_states, settings.PREDICTION_BATCH_SIZE)

            # print("Model predict en train")

        # print("Después de model.predict")
        new_current_states = np.array([transition[3] for transition in minibatch])

        with self.graph.as_default():
            future_qs_list = self.target_model.predict(new_current_states, settings.PREDICTION_BATCH_SIZE)

        X = []
        y = []

        for index, (state_train, action, reward, new_state_train, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + settings.DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(state_train)
            y.append(current_qs)


        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        with self.graph.as_default():
            if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or\
                    settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1] or\
                    settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7] or\
                    settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8] or \
                    settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[9]:
                self.model.fit(np.array(X), np.array(y), batch_size=settings.TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)
            else:
                self.model.fit(np.array(X) / 255, np.array(y), batch_size=settings.TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)


        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > settings.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        #print(state)
        if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or \
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1] or \
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7] or\
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8] or \
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[9]:
            return self.model.predict(np.array(state).reshape(-1, *state.shape))
        else:
            # if settings.IM_LAYERS == 3:
            return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]



    def train_in_loop(self):
        # print("Entra a train in loop")
        # X = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, IM_LAYERS)).astype(np.float32)
        if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or\
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1] or \
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7] or \
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8] or \
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[9]:
            X = np.random.uniform(size=(1, settings.state_dim)).astype(np.float32)
        else:
            X = np.random.uniform(size=(1, settings.IM_HEIGHT_CNN, settings.IM_WIDTH_CNN, settings.IM_LAYERS)).astype(np.float32)

        y = np.random.uniform(size=(1, settings.N_actions)).astype(np.float32)
        with self.graph.as_default():
            # print("Entra a graph")
            print(X.shape)
            self.model.fit(X, y, verbose=False, batch_size=1)


            # print("Model fit en train_in_loop")
        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)


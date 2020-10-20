import random
from collections import deque
import numpy as np
import cv2
import time
import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from keras.models import load_model
from train_DQN import CarEnv
import sympy as sym
import math
import carla_config as settings



if __name__ == '__main__':
    # Memory fraction
    print('entra al main')
    gpu_options = tf.GPUOptions(allow_growth=True)
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
    #print("INTRODUCE EL NOMBRE DEL MODELO (Tipo /home/robesafe/carla/PythonAPI/examples/models/XXXXX.model")
    #MODEL_PATH = input()
    # Load the model
    model = load_model(settings.MODEL_PATH)
    print(settings.MODEL_PATH)
    # Create environment
    env = CarEnv()

    # For agent speed measurements - keeps last 60 frametimes
    fps_counter = deque(maxlen=60)

    # Initialize predictions - first prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps

    if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1] or\
            settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7] or \
            settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8] or \
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[9]:
        aux = np.ones(settings.state_dim, )
        model.predict(np.array(aux).reshape(-1, *aux.shape))
    else:
        aux = np.ones((settings.IM_HEIGHT_CNN, settings.IM_WIDTH_CNN, settings.IM_LAYERS))
        if settings.USE_RNN == 0:
            model.predict(np.array(aux).reshape(-1, *aux.shape) / 255)[0]
        else:
            # state = np.expand_dims(state, -1)
            aux2 = np.ones(settings.N_data_RNN, )
            model.predict([np.array(aux).reshape(-1, *aux.shape) / 255, np.array(aux2).reshape(-1, *aux2.shape)])[0]

    # Loop over episodes
    episode = 0
    while True:

        print('Restarting episode')

        # Reset environment and get initial state

        if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or \
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1] or \
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8] or \
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[9]:
            _, state_train = env.reset()
        else:
            state_train, _ = env.reset()
            data_RNN = np.array([env.trackpos_rw, env.angle_rw])
            if settings.IM_LAYERS == 1:
                state_train = np.expand_dims(state_train, -1)
            if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7]:
                state_train = state_train.flatten()


        env.collision_hist = []

        done = False
        # Guardar los Waypoints
        if settings.GUARDAR_DATOS == 1:
            np.savetxt('Waypoints/waypoints_' + str(settings.WORKING_MODE) + '/' + str(settings.TRAIN_MODE) + '_waypoints' + str(episode) + '.txt', env.waypoints_txt, delimiter=';')
        # Loop over steps
        while True:

            # For FPS counter
            step_start = time.time()

            # Show current frame
            #cv2.imshow(f'Agent - preview', current_state)
            #cv2.waitKey(1)

            # Predict an action based on current observation space
            # qs = model.predict(np.array(state_train).reshape(-1, *state_train.shape))

            if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or settings.WORKING_MODE == \
                    settings.WORKING_MODE_OPTIONS[1] or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7] or \
                    settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8] or \
                    settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[9]:
                qs = model.predict(np.array(state_train).reshape(-1, *state_train.shape))
            else:
                if settings.USE_RNN == 0:
                    qs = model.predict(np.array(state_train).reshape(-1, *state_train.shape) / 255)[0]
                else:
                    qs = model.predict([np.array(state_train).reshape(-1, *state_train.shape) / 255, np.array(data_RNN).reshape(-1, *data_RNN.shape)])[0]



            #print(qs)
            #qs *= [0.975, 1, 0.92]
            action = np.argmax(qs)
            print(settings.ACTIONS_NAMES[action])
            # Step environment (additional flag informs environment to not break an episode by time limit)

            new_data_RNN = np.array([env.trackpos_rw, env.angle_rw])

            if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1] or \
                    settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8] or \
                    settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[9]:
                [_, new_state_train], reward, done, _ = env.step(action)
            else:
                [new_state_train, _], reward, done, _ = env.step(action)

                if settings.IM_LAYERS == 1:
                    new_state_train = np.expand_dims(new_state_train, -1)
                if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7]:
                    new_state_train = new_state_train.flatten()


            # Set current step for next loop iteration
            state_train = new_state_train
            data_RNN = new_data_RNN
            # If done - agent crashed, break an episode
            if done:
                break

            # Measure step time, append to a deque, then print mean FPS for last 60 frames, q values and taken action
            frame_time = time.time() - step_start
            fps_counter.append(frame_time)
            #print(f'Agent: {len(fps_counter)/sum(fps_counter):>4.1f} FPS | Action: [{qs[0]:>5.2f}, {qs[1]:>5.2f}, {qs[2]:>5.2f}] {action}')


        # Guardar la trayectoria
        if settings.GUARDAR_DATOS == 1:
            np.savetxt('Trayectorias/trayectoria_' + str(settings.WORKING_MODE) + '/' + str(settings.TRAIN_MODE) + '_trayectoria_' + str(episode) + '.txt', env.position_array, delimiter=';')
            env.position_array = []
        # Destroy an actor at end of episode
        episode += 1
        for actor in env.actor_list:
            actor.destroy()



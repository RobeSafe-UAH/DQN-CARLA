import os
import sys
import random
import time
import numpy as np
import cv2
import math
from datetime import date


import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from threading import Thread

from tqdm import tqdm


import carla_config as settings
from agent_model import DQNAgent
from carla_env import CarEnv


# Own Tensorboard class

if __name__ == '__main__':
    distance_acum = []
    epsilon = settings.epsilon
    FPS = 60
    # For stats
    ep_rewards = [-200]
    # tf.config.optimizer.set_jit(True)
    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.set_random_seed(1)
    # Memory fraction, used mostly when trai8ning multiple agents
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    gpu_options = tf.GPUOptions(allow_growth=True)
    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')
    # print("Antes de create agent")
    # Create agent and environment
    agent = DQNAgent()
    env = CarEnv()
    # print("Despues de agente y environment")
    date_title = date.today()
    # Start training thread and wait for training to be initialized
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()

    while not agent.training_initialized:
        # print("Esperando inicializacion de agente")
        time.sleep(0.01)
    # print("Antes de get_qs")

    # Initialize predictions - forst prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    # agent.get_qs(np.ones((env.im_height, env.im_width, IM_LAYERS)))

    if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or \
            settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1] or \
            settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7] or \
            settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8] or \
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[9]:
        agent.get_qs(np.ones(settings.state_dim, ))
    else:
        agent.get_qs(np.ones((settings.IM_HEIGHT_CNN, settings.IM_WIDTH_CNN, settings.IM_LAYERS)))


    # Iterate over episodes
    for episode in tqdm(range(1, settings.EPISODES + 1), ascii=True, unit='episodes'):
        # try:
        env.collision_hist = []
        env.crossline_hist = []

        # Update tensorboard step every episode
        agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or\
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1] or\
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8] or \
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[9]:
            _, state_train = env.reset()
        else:
            state_train, _ = env.reset()
            if settings.IM_LAYERS == 1:
                state_train = np.expand_dims(state_train, -1)
            if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7]:
                state_train = state_train.flatten()


        done = False
        episode_start = time.time()
        # Play for given number of seconds only

        while True:

            # This part stays mostly the same, the change is to query a model for Q values
            if np.random.random() > epsilon:
                # Get action from Q table
                action_vector = agent.get_qs(state_train)
                action = np.argmax(action_vector)
                #print(settings.ACTIONS_NAMES[action])
                # print(settings.ACTIONS_NAMES[action])

            else:
                # Get random action
                # print("Accion random")
                action = np.random.randint(0, settings.N_actions)
                # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                time.sleep(1 / FPS)

            # if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0]:
            #     [_, new_state_train], reward, done, _ = env.step(action)
            # elif settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1]:
            #     new_image, reward, done, info = env.step(action)
            #     new_state_train = env.Calcular_estado(new_image)

            if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or \
                    settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1] or\
                    settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8] or \
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[9]:
                [_, new_state_train], reward, done, _ = env.step(action)
            else:
                [new_state_train, _], reward, done, _ = env.step(action)
                if settings.IM_LAYERS == 1:
                    new_state_train = np.expand_dims(new_state_train, -1)
                if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7]:
                    new_state_train = new_state_train.flatten()

            #print('Action: ', ACTIONS_NAMES[action], ' Reward: ', reward)

            # Transform new continous state to new discrete state and count reward
            episode_reward += reward

            # Every step we update replay memory
            agent.update_replay_memory((state_train, action, reward, new_state_train, done))


            state_train = new_state_train
            step += 1
            if done:
                break


        #print(agent.model.get_weights())
        #json_wei = agent.model.to_json()
        #print(json_wei)


        # End of episode - destroy agents
        for actor in env.actor_list:
            actor.destroy()


        # Decay epsilon
        if epsilon > settings.MIN_EPSILON:
            epsilon *= settings.EPSILON_DECAY
            epsilon = max(settings.MIN_EPSILON, epsilon)

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if (episode > 1) and ((episode % settings.AGGREGATE_STATS_EVERY) == 0) or (episode == 2):
            average_reward = sum(ep_rewards[-settings.AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-settings.AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-settings.AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-settings.AGGREGATE_STATS_EVERY:])
            average_dist = sum(env.distance_acum[-settings.AGGREGATE_STATS_EVERY:]) / len(env.distance_acum[-settings.AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                           efshowpsilon=epsilon, avegare_dist=average_dist)

        #Guardar datos del entrenamiento en ficheros
        if episode % 3 == 0:
            agent.model.save(settings.AGENT_PATH + str(settings.TRAIN_MODE)+"_model.model")
        if episode % settings.N_save_stats == 0:
            agent.model.save(settings.AGENT_PATH + str(settings.TRAIN_MODE)+"_" + str(episode) + "_model.model")
        if (episode > 10) and (episode_reward > np.max(ep_rewards[:-1])):
            agent.model.save(settings.AGENT_PATH + str(settings.TRAIN_MODE)+"_best_reward_model.model")

        acum = 0

    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    trainer_thread.join()
    agent.model.save(settings.AGENT_PATH + str(settings.TRAIN_MODE) + "_last_model.model")
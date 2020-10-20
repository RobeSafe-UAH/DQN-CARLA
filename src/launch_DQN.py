import os
import carla_config as settings
import time

#Clear Carla Environment
print('### Reseting Carla Map ###')
os.system('python3 ' + settings.path2CARLA + 'PythonAPI/util/config.py -m ' + str(settings.CARLA_MAP))
time.sleep(5)


print('####### RUNNING DQN', settings.WORKING_MODE, ' IN ', settings.TRAIN_PLAY[settings.TRAIN_PLAY_MODE], ' MODE #######')

if settings.TRAIN_PLAY_MODE == 1:
    os.system('python3 train_DQN.py')
elif settings.TRAIN_PLAY_MODE == 0:
    os.system('python3 play_DQN.py')

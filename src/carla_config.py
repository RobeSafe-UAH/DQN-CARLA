
#TRAINING MODES
TRAIN_PLAY_MODE = 0  # Train=1 Play=0
GUARDAR_DATOS = 0 # 1 Para guardar trayectoria recorrida y waypoints
TRAIN_PLAY = ["PLAY", "TRAIN"]
WORKING_MODE_OPTIONS = ["WAYPOINTS_CARLA","WAYPOINTS_IMAGE","CNN_BW_TRAJECTORY","CNN_RGB","CNN_RGB_TRAJECTORY",
                        "CNN_SEMANTIC", "CNN_GRAYSCALE", "CNN_FLATTEN", "TP_ANG", "PRE_TRAINED_CNN"]

WORKING_MODE = "PRE_TRAINED_CNN"

#TRAINING STAGES
CARLA_MAP = 'Town01'   # "mapa_oscar_v5"
TRAIN_MODE_OPTIONS=["RANDOM", "STRAIGHT", "TURN_LEFT", "TURN_RIGHT", "TURN_RIGHT_LEFT", "TURN_LEFT_RIGHT", "ALTERNATIVE"]
TRAIN_MODE = "TURN_RIGHT_LEFT"

path2CARLA = "/home/robesafe/carla/" # PATH hasta carla se utiliza para limpiar el mapa
#path2carla = "/home/proyectosros/carla/carla/"

#IMAGE CONFIGURATION
IM_WIDTH_VISUALIZATION = 640*2
IM_HEIGHT_VISUALIZATION = 480
IM_WIDTH_CNN = 160 #160 #PARA MANTENER LA RELACION DE 8/3 DEBERÍA SER 160
IM_HEIGHT_CNN = 60 #60
tau = 0.001  # Target Network HyperParameter
lra = 0.0001  # Learning rate for Actor
lrc = 0.001  # Learning rate for Critic
episodes_num = 3500
max_steps = 100000
buffer_size = 100000
batch_size = 32
gamma = 0.99  # discount factor
hidden_units = (300, 600)
SECONDS_PER_EPISODE = 70
SHOW_CAM = 1
SHOW_WAYPOINTS = 1
SHOW_CAM_RESIZE = 0

########################   ADD FOR DQN   #####################
ACTIONS_NAMES = {
    0: 'forward_slow',
#    1: 'forward_medium',
    1: 'left_slow',
    2: 'left_medium',
    3: 'right_slow',
    4: 'right_medium',
#    6: 'brake_light',
    #3: 'no_action',
}
N_actions = len(ACTIONS_NAMES)

ACTION_CONTROL = {
    0: [0.55, 0, 0],
    # 1: [0.7, 0, 0],
    1: [0.4, 0, -0.1],
    2: [0.4, 0, -0.4],
    3: [0.4, 0, 0.1],
    4: [0.4, 0, 0.4],
    # 6: [0, 0.3, 0],
    #3: None,
}

modo_recompensa = 2
STEER_AMT = 0.2
CNN_MODEL = 2
DISCOUNT = 0.99
if TRAIN_PLAY_MODE == 1:
    epsilon = 1
elif TRAIN_PLAY_MODE == 0:
    epsilon = 0

EPSILON_DECAY = 0.9995
MIN_EPSILON = 0.01
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 4
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 8
UPDATE_TARGET_EVERY = 5
EPISODES = 9_000
AGGREGATE_STATS_EVERY = 10
AGENT_PATH = "../models/data_" + str(WORKING_MODE) + "/"
# MODEL_PATH = "models/data_" + str(WORKING_MODE) + "/" + str(TRAIN_MODE) + "_best_reward_model.model"
# MODEL_PATH = "../models/data_" + str(WORKING_MODE) + "/RANDOM_8900_model.model"
MODEL_PATH = "../models/data_WAYPOINTS_CARLA/RANDOM_8900_model.model"

# MODEL_PATH = "../models/data_" + str(WORKING_MODE) + "/ALTERNATIVE_7300_model.model"
# MODEL_PATH = "../models/data_" + str(WORKING_MODE) + "/TURN_RIGHT_best_reward_model.model"
# MODEL_PATH = "models/data_PRE_TRAINED_CNN/TURN_RIGHT_best_reward_model.model"
PRE_CNN_PATH = "../PRE_CNN_models/PilotNet_2002m_BEV.model"

N_save_stats = 100
########################   ADD FOR DQN   #####################

#Vista de pájaro
BEV_PRE_CNN = 0

#WORKING TYPE SELECTION
WAYPOINTS = 'X'     # X para utilizar solo coordenadas X, XY para utilizar coordenadas XY
THRESHOLD = 0  # FLAG DE UMBRALIZACIÓN
DRAW_TRAJECTORY = 0 # NO PINTAR=0, PINTAR=1
IM_LAYERS = 1
state_dim = 16 #Dimension de los datos de entrada a la red.
dimension_vector_estado = 16 #Dimension del vector de estado calculado en transform2lcoal, necesario para recompensa en todos los casos
if WORKING_MODE == WORKING_MODE_OPTIONS[0]:         # WAYPOINTS_CARLA
    if WAYPOINTS == 'XY':
        state_dim = 31
        dimension_vector_estado = state_dim
    elif WAYPOINTS == 'X':
        state_dim = 16
    CAM_X = 1.0
    CAM_Z = 1.8
    CAM_PITCH = -20.0
    CAM_YAW = 0.0
    CAM_ROLL = 0.0
elif WORKING_MODE == WORKING_MODE_OPTIONS[1]:       # WAYPOINTS_IMAGE
    DRAW_TRAJECTORY = 1
    state_dim = 17
elif WORKING_MODE == WORKING_MODE_OPTIONS[2]:       # CNN_BW_TRAJECTORY
    IM_LAYERS = 1                               # GrayScale = 1, Color = 3
    IM_TYPE = 0                                 # RGB=0, SemanticSegmetnation=1
    THRESHOLD = 1                               # FLAG DE UMBRALIZACIÓN
    DRAW_TRAJECTORY = 1
elif WORKING_MODE == WORKING_MODE_OPTIONS[3]:       # CNN_RGB
    IM_LAYERS = 3                               # GrayScale = 1, Color = 3
    IM_TYPE = 0                                 # RGB=0, SemanticSegmetnation=1
elif WORKING_MODE == WORKING_MODE_OPTIONS[4]:       # CNN_RGB_TRAJECTORY
    IM_LAYERS = 3                               # GrayScale = 1, Color = 3
    IM_TYPE = 0                                 # RGB=0, SemanticSegmetnation=1
    DRAW_TRAJECTORY = 1
elif WORKING_MODE == WORKING_MODE_OPTIONS[5]:       # CNN_SEMANTIC
    IM_LAYERS = 3                               # GrayScale = 1, Color = 3
    IM_TYPE = 1                                 # RGB=0, SemanticSegmetnation=1
elif WORKING_MODE == WORKING_MODE_OPTIONS[6]:       # CNN_GRAYSCALE
    IM_LAYERS = 1                               # GrayScale = 1, Color = 3
    IM_TYPE = 0                                 # RGB=0, SemanticSegmetnation=1
elif WORKING_MODE == WORKING_MODE_OPTIONS[7]:       # CNN_FLATTEN
    IM_LAYERS = 1                               # GrayScale = 1, Color = 3
    IM_TYPE = 0                                 # RGB=0, SemanticSegmetnation=1
    THRESHOLD = 1                               # FLAG DE UMBRALIZACIÓN
    DRAW_TRAJECTORY = 1
    state_dim = IM_WIDTH_CNN * IM_HEIGHT_CNN

elif WORKING_MODE == WORKING_MODE_OPTIONS[8]:       # TRACKPOS_ANGLE
    state_dim = 2

elif WORKING_MODE == WORKING_MODE_OPTIONS[9]:       # PRE-TAINED-CNN
    if WAYPOINTS == 'XY':
        state_dim = 31
        dimension_vector_estado = state_dim
    elif WAYPOINTS == 'X':
        state_dim = 16
    IM_LAYERS = 3                               # GrayScale = 1, Color = 3
    IM_TYPE = 0                                 # RGB=0, SemanticSegmetnation=1
    DRAW_TRAJECTORY = 1
    THRESHOLD = 0                               # FLAG DE UMBRALIZACIÓN
    IM_WIDTH_VISUALIZATION = 2*640
    CAM_X = 1.0
    CAM_Z = 2
    CAM_PITCH = -20.0
    CAM_YAW = 0.0
    CAM_ROLL = 0.0
    BEV_PRE_CNN = 1
"""This file contains all the method input values at a single place"""

import datetime

# Training Inputs
EPOCHS = 50
VAL_SUB_SPLIT = 5
BUFFER_SIZE = 500
BATCH_SIZE = 32
VAL_SPLIT = 0.2

# Model Inputs
FILTERS = 32
CLASSES = 23
INPUT_SIZE = ([32, 96, 128, 3])
INF_INPUT_SIZE = (1, 96, 128, 3)

# Callbacks Inputs
CURR_DATETIME = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
MODEL_DIR = "./saved_model/"
MODEL_EXTENSION = '.hdf5'
#MODEL_FILEPATH = MODEL_DIR + CURR_DATETIME + '.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5'
#MODEL_FILEPATH = MODEL_DIR + CURR_DATETIME + MODEL_EXTENSION
TENSORBOARD_LOG_DIR = "./tensorboard_logs_dir/logs"+CURR_DATETIME
LOGGER_DIR = "./csv_logger_dir/training"+CURR_DATETIME+".log"
SAVE_WEIGHTS_ONLY = True
SAVE_BEST_ONLY = True

# Data Inputs
IMAGES_SRC = "./data/carla/images"
MASKS_SRC = "./data/carla/masks"

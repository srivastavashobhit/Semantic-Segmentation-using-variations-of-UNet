import tensorflow as tf
import glob
import os

from tensorflow.python.keras.callbacks import CSVLogger

from model_unet_tced import UNetTCED
from utils.values_utils import LOGGER_DIR, TENSORBOARD_LOG_DIR, CLASSES, \
    FILTERS, INPUT_SIZE, CURR_DATETIME, MODEL_DIR, MODEL_EXTENSION


def get_callbacks(model_type, ckpt_dir=MODEL_DIR, ckpt_datetime=CURR_DATETIME, ckpt_extension=MODEL_EXTENSION,
                  logger_dir=LOGGER_DIR, tensorboard_dir=TENSORBOARD_LOG_DIR):
    ckpt_path = ckpt_dir + ckpt_datetime + '_' + model_type + ckpt_extension
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path,
                                                    save_weights_only=True,
                                                    monitor='val_accuracy',
                                                    mode='max',
                                                    save_best_only=True)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                     patience=3)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                      patience=3)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir)

    csv_logger = CSVLogger(logger_dir)

    callbacks = [checkpoint, reduce_lr, early_stopping, tensorboard, csv_logger]

    return callbacks


def get_latest_model(model_type):
    model = UNetTCED(FILTERS, CLASSES, INPUT_SIZE)
    all_models_names = [os.path.basename(x) for x in glob.glob(os.path.join(MODEL_DIR, "*.hdf5"))]
    all_models_names_sliced = [x[:15] for x in all_models_names]
    latest_ts = max(all_models_names_sliced)
    latest_ts_ind = all_models_names_sliced.index(latest_ts)
    latest_model_name = all_models_names[latest_ts_ind]
    latest_model = os.path.join(MODEL_DIR, latest_model_name)
    model.load_weights(latest_model)
    return model


def generate_prediction(model, input_image):
    prediction = model.predict(input_image)
    return prediction

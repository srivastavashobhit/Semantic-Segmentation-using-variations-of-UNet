import tensorflow as tf
import glob
import os

from tensorflow.python.keras.callbacks import CSVLogger

from model_unet_tced import UNetTCED
from utils.values_utils import MODEL_DIR, MODEL_FILEPATH, SAVE_WEIGHTS_ONLY, LOGGER_DIR, TENSORBOARD_LOG_DIR, CLASSES, \
    FILTERS, INPUT_SIZE


def get_callbacks(ckpt_path=MODEL_FILEPATH, save_weights_only=SAVE_WEIGHTS_ONLY, logger_dir=LOGGER_DIR,
                  tensorboard_dir=TENSORBOARD_LOG_DIR):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path,
                                                    save_weights_only=save_weights_only,
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


def get_model_from_checkpoint():

    model = UNetTCED(FILTERS, CLASSES, INPUT_SIZE)
    all_models_names = [os.path.basename(x) for x in glob.glob(os.path.join(MODEL_DIR, "*.hdf5"))]
    latest = os.path.join(MODEL_DIR, max(all_models_names))
    model.load_weights(latest)
    return model


def generate_prediction(model, input_image):
    prediction = model.predict(input_image)
    return prediction

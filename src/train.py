from model_unet_tced import UNetTCED
from model_unet_std import UNetSTD
from utils.data_utils import get_train_dataset
from utils.model_utils import get_callbacks, get_latest_model
from utils.values_utils import IMAGES_SRC, MASKS_SRC, VAL_SPLIT, BATCH_SIZE, EPOCHS, FILTERS, CLASSES, INPUT_SIZE


def train_new_model(model_type, images_src=IMAGES_SRC, masks_src=MASKS_SRC, val_split=VAL_SPLIT, batch_size=BATCH_SIZE):
    train_dataset, val_dataset = get_train_dataset(images_src, masks_src, val_split, batch_size)
    if model_type == 'UNetTCED':
        print("Model: UNet Tightly Connected Encoder and Decoder")
        model = UNetTCED(FILTERS, CLASSES, INPUT_SIZE)
    else:
        print("Model: Standard UNet")
        model = UNetSTD(FILTERS, CLASSES, INPUT_SIZE)

    print(model.summary())
    model.fit(x=train_dataset,
              epochs=EPOCHS,
              validation_data=val_dataset,
              callbacks=get_callbacks(model_type))


def train_from_ckpt(model_type, images_src=IMAGES_SRC, val_src=MASKS_SRC, val_split=VAL_SPLIT, batch_size=BATCH_SIZE):
    train_dataset, val_dataset = get_train_dataset(images_src, val_src, val_split, batch_size)
    model = get_latest_model(model_type)
    print(model.summary)
    model.fit(x=train_dataset,
              epochs=EPOCHS,
              validation_data=val_dataset,
              callbacks=get_callbacks(model_type))

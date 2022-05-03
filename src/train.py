from utils.data_utils import *
from utils.model_utils import *
from utils.values_utils import IMAGES_SRC, MASKS_SRC, VAL_SPLIT, BATCH_SIZE, EPOCHS


def train_new_model(images_src=IMAGES_SRC, masks_src=MASKS_SRC, val_split=VAL_SPLIT, batch_size=BATCH_SIZE):
    train_dataset, val_dataset = get_train_dataset(images_src, masks_src, val_split, batch_size)
    model = UNet(FILTERS, CLASSES, INPUT_SIZE)
    print("TDL From Model New")
    print(model.summary())
    model.fit(x=train_dataset,
                        epochs=EPOCHS,
                        validation_data=val_dataset,
                        callbacks=get_callbacks())

def train_from_ckpt(images_src=IMAGES_SRC, val_src=MASKS_SRC, val_split=VAL_SPLIT, batch_size=BATCH_SIZE):
    train_dataset, val_dataset = get_train_dataset(images_src, val_src, val_split, batch_size)
    model = get_model_from_checkpoint()
    print("TDL From Model Checkpoint")
    print(model.summary)
    model.fit(x=train_dataset,
                        epochs=EPOCHS,
                        validation_data=val_dataset,
                        callbacks=get_callbacks())


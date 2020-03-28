import math
from typing import *
from tensorflow.keras.callbacks import ModelCheckpoint, TerminateOnNaN, LearningRateScheduler, \
    CSVLogger, ReduceLROnPlateau, EarlyStopping
from datetime import datetime
import os

now = datetime.now().strftime("%d-%m-%Y:%H")


def existsfolder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def callbacks(save_path: str, depth: int) -> List:
    """Keras callbacks which include ModelCheckpoint, CSVLogger, TensorBoard, LearningRateScheduler, TerminateOnNaN
    
    Parameters
    ----------
    save_path: str
        local directory to save model weights
    depth : int
        Depth of ResNet model

    Returns
    -------
    List
        List all callbacks
    """
    existsfolder(save_path)

    model_checkpoint = ModelCheckpoint(
        filepath=f"{save_path}/" + f"ResNet{depth}" + "-epoch:{epoch:02d}-val_acc:{val_accuracy:.2f}.hdf5",
        save_best_only=True,
        save_weights_only=False,
        verbose=1)

    existsfolder('./assets/logs')

    csv_logger = CSVLogger(filename=f"./assets/logs/logs-{now}.csv",
                           append=True)

    def lr_schedule(epoch):
        if epoch < 10:
            return 0.003
        elif epoch < 50:
            return 0.0003
        else:
            return 0.00003

    lr_reduce = ReduceLROnPlateau(
        monitor='val_loss',
        factor=math.sqrt(0.1),
        patience=5,
        min_lr=3e-6,
        verbose=1
    )

    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)

    early = EarlyStopping(
        monitor='val_loss',
        min_delta=1e-4,
        patience=15
    )

    terminate_on_nan = TerminateOnNaN()

    callbacks_list = [csv_logger, lr_scheduler, lr_reduce, early, model_checkpoint, terminate_on_nan]
    return callbacks_list

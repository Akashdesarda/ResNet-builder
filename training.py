from tensorflow.keras.optimizers import Adam

from core.data_loader import DataLoader
from core.resnet import build_resnet_model
from utils.callbacks import callbacks
from utils.misc_utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

depth = 56
batch_size = 32
epochs = 50
# TF GPU memory graph
limit_gpu()
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#             logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#             print('[INFO]... ',len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)

dl = DataLoader()

train_generator,xtest,ytest = dl.from_common_dir(
    directory='/home/akash/project/Dataset/cifar10 (1)/train/',
    target_size=(32, 32),
    batch_size=batch_size
)

model = build_resnet_model(
    input_shape=(32, 32, 3),
    depth=depth,
    num_classes=dl.num_classes
)

callbacks = callbacks(
    save_path='./assets/weights/exp2',
    depth=depth
)

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(amsgrad=True, decay=0.001/epochs),
    metrics=['accuracy']
)

history = model.fit(
    x=train_generator,
    epochs=epochs,
    steps_per_epoch=int(dl.train_len/batch_size),
    callbacks=callbacks,
    validation_data=(xtest, ytest),
    validation_steps=int(dl.val_len/batch_size)
)

visualize(
    history=history.history,
    save_dir='./assets/logs'
)

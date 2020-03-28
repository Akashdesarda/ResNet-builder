import numpy as np
from tqdm import tqdm
from imutils import paths
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
import tensorflow as tf

image_paths = list(paths.list_images(''))
data = np.array(np.array(load_img(image_path, target_size=(), color_mode=)) for image_path in image_paths)

model = load_model()
predictions = model.predict(data)
pred_proba = np.argmax(predictions, axis=1)
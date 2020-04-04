# Input will be classwise folders. Perform Load --> Augmentation --> ImageDataGenerator. Return Train generator, validation generator
import os
from typing import *

import numpy as np
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm


class DataLoader:
    """
    API to load and Augment Image Data
    """

    def __init__(self,
                 featurewise_center: bool = False,
                 featurewise_std_normalization: bool = False,
                 zca_whitening: bool = True,
                 rotation_range: int = 30,
                 width_shift_range: float = 0.2,
                 height_shift_range: float = 0.2,
                 zoom_range: float = 0.15,
                 shear_range: float = 0.15,
                 horizontal_flip: bool = True,
                 validation_split: float = 0.3):
        """
        Generate batches of tensor image data with real-time data augmentation.
        The data will be looped over (in batches).

        Parameters
        ----------
        featurewise_center : bool, optional
            Set input mean to 0 over the dataset, feature-wise, by default True
        featurewise_std_normalization : bool, optional
            Divide inputs by std of the dataset, feature-wise, by default True
        zca_whitening : bool, optional
            Apply ZCA whitening, by default True
        rotation_range : int, optional
            Degree range for random rotations, by default 30
        width_shift_range : float, optional
            Shifts width, by default o.2
        height_shift_range : float, optional
            Shifts height, by default o.2
        zoom_range : float, optional
            Range for random zoom, by default o.15
        shear_range : float, optional
            Shear Intensity (Shear angle in counter-clockwise direction in degrees), by default o.15
        horizontal_flip : bool, optional
            Randomly flip inputs horizontally, by default True
        validation_split: float, optional
            Split data for training and validation data, by default 0.3
        """
        self.featurewise_center = featurewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.zca_whitening = zca_whitening
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.zoom_range = zoom_range
        self.shear_range = shear_range
        self.horizontal_flip = horizontal_flip
        self.validation_split = validation_split
        self.train_len = None
        self.val_len = None
        self.classes = None
        self.num_classes = None

    # @staticmethod
    def _datagen(self):
        return ImageDataGenerator(
            # rescale=1. / 255,
            featurewise_center=self.featurewise_center,
            featurewise_std_normalization=self.featurewise_std_normalization,
            zca_whitening=self.zca_whitening,
            rotation_range=self.rotation_range,
            width_shift_range=self.width_shift_range,
            height_shift_range=self.height_shift_range,
            zoom_range=self.zoom_range,
            shear_range=self.shear_range,
            horizontal_flip=self.horizontal_flip,
            validation_split=self.validation_split
        )

    # def from_common_dir(self, path: str,
    #                     target_size: tuple = (256, 256),
    #                     color_mode: str = 'rgb',
    #                     classes: int = None,
    #                     class_mode: str = 'categorical',
    #                     batch_size: int = 32):
    #     """
    #     Takes the path to a directory & generates batches of augmented data. Use this only if directory is common for
    #     both training data and validation data.
    #
    #     Notes
    #     -----
    #     Following should be the folder structure
    #     Dir/sub_dir/*.jpeg --> Dir/class/*.jpeg
    #
    #     Parameters
    #     ----------
    #     path : str
    #         Path of Image Directory. It should contain one subdirectory per class.
    #         Any PNG, JPG, BMP, PPM or TIF images inside each of the subdirectories
    #         directory tree will be included in the generator.
    #     target_size : tuple, optional
    #         The dimensions to which all images found will be resized, by default (256,256)
    #     color_mode : str, optional
    #         One of "grayscale", "rgb", "rgba".. Whether the images will be converted to have 1, 3, or 4 channels,
    #         by default "rgb"
    #     classes : list, optional
    #         list of class subdirectories (e.g. ['dogs', 'cats']). If not provided, the list of classes will be
    #         automatically inferred from the subdirectory names/structure under directory,
    #         where each subdirectory will be treated as a different class,
    #         by default None
    #     class_mode : str, optional
    #         One of "categorical", "binary", "sparse", "input", or None.
    #         Determines the type of label arrays that are returned, by default "categorical"
    #     batch_size :
    #
    #     Returns
    #     -------
    #     train_generator :
    #         Data Loader for training data
    #     val_generator:
    #         Data loader for validation data
    #
    #     References
    #     -------
    #     Visit: https://keras.io/preprocessing/image/#flow_from_directory
    #
    #     """
    #     # Training Data Generator
    #     train_generator = self._datagen().flow_from_directory(
    #         directory=path,
    #         target_size=target_size,
    #         color_mode=color_mode,
    #         classes=classes,
    #         class_mode=class_mode,
    #         batch_size=batch_size,
    #         subset='training'
    #     )
    #
    #     # Validation Data Generator
    #     val_generator = self._datagen().flow_from_directory(
    #         directory=path,
    #         target_size=target_size,
    #         color_mode=color_mode,
    #         classes=classes,
    #         class_mode=class_mode,
    #         batch_size=batch_size,
    #         subset='validation'
    #     )
    #     return train_generator, val_generator

    def from_dir(self, directory: str,
                 target_size: tuple = (256, 256),
                 color_mode: str = 'rgb',
                 classes: int = None,
                 class_mode: str = 'categorical',
                 batch_size: int = 32):
        """
        Takes the path to a directory & generates batches of augmented data. Use this only if directory is separate for
        both training data and validation data. Therefore use this method to create training as well as validation
        data generator separately.

        Parameters
        ----------
        directory : str
            Path of Image Directory. It should contain one subdirectory per class.
            Any PNG, JPG, BMP, PPM or TIF images inside each of the subdirectories
            directory tree will be included in the generator.
        target_size : tuple, optional
            The dimensions to which all images found will be resized, by default (256,256)
        color_mode : str, optional
            One of "grayscale", "rgb", "rgba".. Whether the images will be converted to have 1, 3, or 4 channels,
            by default "rgb"
        classes : list, optional
            list of class subdirectories (e.g. ['dogs', 'cats']). If not provided, the list of classes will be
            automatically inferred from the subdirectory names/structure under directory,
            where each subdirectory will be treated as a different class,
            by default None
        class_mode : str, optional
            One of "categorical", "binary", "sparse", "input", or None.
            Determines the type of label arrays that are returned, by default "categorical"
        batch_size : int, optional
            Size of the batches of data, by default 32

        Returns
        -------
        data_generator :
            Data Loader for given data

        Notes
        -----
        Following should be the folder structure
        Dir/sub_dir/*.jpeg --> Dir/class/*.jpeg

        References
        -------
        Visit: https://keras.io/preprocessing/image/#flow_from_directory

        """
        return self._datagen().flow_from_directory(
            directory=directory,
            target_size=target_size,
            color_mode=color_mode,
            classes=classes,
            class_mode=class_mode,
            batch_size=batch_size
        )

    def from_common_dir(self, directory: str, target_size: Tuple[int, int], color_mode: str = 'rgb', batch_size: int = 32):
        """
        Takes the path to a directory & generates batches of augmented data. Use this only if directory is common for
        both training data and validation data.

        Notes
        -----
        Following should be the folder structure
        Dir/sub_dir/*.jpeg --> Dir/class/*.jpeg

        Parameters
        ----------
        directory : str
            Path of Image Directory. It should contain one subdirectory per class.
            Any PNG, JPG, BMP, PPM or TIF images inside each of the subdirectories
            directory tree will be included in the generator.
        target_size : tuple, optional
            The dimensions to which all images found will be resized, by default (256,256)
        color_mode : str, optional
            One of "grayscale", "rgb", "rgba".. Whether the images will be converted to have 1, 3, or 4 channels,
            by default "rgb"
        batch_size : int, optional
            Size of batches of images to be used, by default 32

        Returns
        -------
        ImageGenerator : Data generator to be used in fit_generator() Keras API
        x_test: Validation data
        y_test: Validation labels

        References
        -------
        Visit: https://keras.io/preprocessing/image/#flow

        """
        image_paths = list(paths.list_images(basePath=directory))

        data = np.array([np.array(load_img(image_path, target_size=target_size, color_mode=color_mode)) for image_path in tqdm(image_paths)]) / 255.0
        labels = np.array([image_path.split(os.path.sep)[-2] for image_path in image_paths])
        self.classes, self.num_classes = np.unique(labels).tolist(), len(np.unique(labels))

        encoder = LabelEncoder()
        labels = encoder.fit_transform(labels)
        labels = to_categorical(labels, num_classes=self.num_classes)

        xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size=self.validation_split, random_state=42)

        self.train_len = len(ytrain)
        self.val_len = len(ytest)

        print(f"[INFO]...List of classes found are: {self.classes}")
        print(f"[INFO]...Training size: {self.train_len} for {self.num_classes} classes")
        print(f"[INFO]...Validation size: {self.val_len} for {self.num_classes} classes")

        train_generator = self._datagen().flow(
            x=xtrain,
            y=ytrain,
            batch_size=batch_size
        )
        return train_generator,xtest,ytest

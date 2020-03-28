import os
import pandas as pd
import seaborn as sns
import tensorflow as tf
from typing import *
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

now = datetime.now().strftime("%d-%m-%Y:%H")

def existsfolder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def limit_gpu():
    # Tf GPU memory graph
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print('[INFO]... ',len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def visualize(history: Dict, save_plot: bool = True, save_dir: str = None):
    """Visualize training history of model
    
    Parameters
    ----------
    history : Dict
        model.fit history
    save_plot : bool, optional
        save plot to hard disk, by default True
    save_dir : str, optional
        path to save plot, by default None
    """
    df = pd.DataFrame(history)
    sns.set_style('whitegrid')
    plt.figure(figsize=(15, 10))
    plt.title('Model Performance')
    plt.xlabel('Epochs #')
    plt.ylabel('Accuracy/Loss')
    sns.lineplot(data=df, markers=True, dashes=False)
    if save_plot is not False:
        existsfolder(f"{save_dir}/logs")
        plt.savefig(f"{save_dir}/logs/model_performance-{now}.png")
    else:
        plt.show()


def report(y_true, y_pred, labels: List = None):
    """Logging of report
    
    Parameters
    ----------
    y_true : numpy array or pandas series
        lables of test data
    y_pred : numpy array or pandas series
        labels of predicted data
    labels : list, optional
        List of classes' labels
    """
    print(f"[REPORT]...Accuracy of recently trained model is: {accuracy_score(y_true, y_pred)}")
    print("Following is detailed classification Report")
    existsfolder('./assets/logs')
    if labels is None:
        print(classification_report(y_true, y_pred))
        with open(f"/assets/logs/report_{now}.txt", "w") as file:
            print(classification_report(y_true, y_pred), file=file)
    else:
        print(classification_report(y_true, y_pred, target_names=labels))
        with open(f"/assets/logs/report_{now}.txt", "w") as file:
            print(classification_report(y_true, y_pred), file=file)

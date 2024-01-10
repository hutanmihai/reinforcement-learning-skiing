import cv2
import numpy as np
from src.dqn.constants import WINDOW_SIZE


def crop(frame: np.ndarray) -> np.ndarray:
    """
    Crops the frame image to the relevant part of the screen.
    :param frame: the frame(state) image
    :return: the cropped image
    """
    # Exact crop [30:180, 8:152]
    # Rounded crop [30:180, 10:150]
    # Maybe try with both of them
    return frame[30:180, 8:152]


def resize(frame: np.ndarray) -> np.ndarray:
    """
    Resizes the frame image.
    :param frame: the frame(state) image
    :return: the resized image
    """
    state = cv2.resize(frame, (WINDOW_SIZE, WINDOW_SIZE))
    return state


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """
    Converts a rgb image array to a grey image array.

    :param rgb: the rgb image array.
    :return: the converted array.
    """
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return grayscale


def format2pytorch(frame: np.ndarray) -> np.ndarray:
    """
    Formats the frame image to be used with PyTorch. It does this by adding a new axis to the image array.
    (int, int) -> (1, int, int) for PyTorch
    :param frame: the frame(state) image
    :return: the formatted image
    """
    return frame[np.newaxis, :, :]


def normalize(frame: np.ndarray) -> np.ndarray:
    """
    Normalizes the frame image.
    :param frame: the frame(state) image
    :return: the normalized image
    """
    frame = frame.astype(np.float32)
    frame /= 255.0
    return frame


def preprocess(frame: np.ndarray) -> np.ndarray:
    """
    Preprocesses the frame image.
    :param frame: the frame(state) image
    :return: the preprocessed image
    """
    frame = crop(frame)
    frame = resize(frame)
    frame = rgb2gray(frame)
    frame = format2pytorch(frame)
    frame = normalize(frame)
    return frame

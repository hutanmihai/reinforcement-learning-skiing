import cv2
import numpy as np


def crop(state: np.ndarray) -> np.ndarray:
    """
    Crops the state image to the relevant part of the screen.
    :param state: the state image
    :return: the cropped image
    """
    # Exact crop [30:180, 8:152]
    # Rounded crop [30:180, 10:150]
    # Maybe try with both of them
    return state[30:180, 10:150]


def resize(state: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Downsamples the state image.
    :param state: the state image
    :param scale: the scale to downsample by
    :return: the downsampled image
    """
    return state[::scale, ::scale]


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """
    Converts an rgb image array to a grey image array.

    :param rgb: the rgb image array.
    :return: the converted array.
    """
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    grayscale = grayscale[np.newaxis, :, :]  # (75, 70) -> (1, 75, 70) for PyTorch
    return grayscale


def preprocess(state: np.ndarray) -> np.ndarray:
    """
    Preprocesses the state image.
    :param state: the state image
    :return: the preprocessed image
    """
    state = crop(state)
    state = resize(state)
    state = rgb2gray(state)
    return state

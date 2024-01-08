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
    return state[30:180, 8:152]


def resize(state: np.ndarray) -> np.ndarray:
    """
    Downsamples the state image.
    :param state: the state image
    :param scale: the scale to downsample by
    :return: the downsampled image
    """
    state = cv2.resize(state, (80, 80))
    return state


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """
    Converts an rgb image array to a grey image array.

    :param rgb: the rgb image array.
    :return: the converted array.
    """
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    grayscale = grayscale[np.newaxis, :, :]  # (75, 70) -> (1, 75, 70) for PyTorch
    return grayscale


def normalize(state: np.ndarray) -> np.ndarray:
    """
    Normalizes the state image.
    :param state: the state image
    :return: the normalized image
    """
    state = state.astype(np.float32)
    state /= 255.0
    return state


def preprocess(state: np.ndarray) -> np.ndarray:
    """
    Preprocesses the state image.
    :param state: the state image
    :return: the preprocessed image
    """
    state = crop(state)
    state = resize(state)
    state = rgb2gray(state)
    state = normalize(state)
    return state

import os
from pathlib import Path

import numpy as np
import cv2 as cv
import plotly.graph_objects as go

from src.dqn.constants import DQN_PERFORMANCE_PATH


def show_image(image: np.ndarray, title: str = "image") -> None:
    """
    Shows the image.
    :param title: the title of the window, default is "image"
    :param image: the image to show
    :return:
    """
    cv.namedWindow(title, cv.WINDOW_KEEPRATIO)
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def check_if_dirs_exist(paths: list[Path] | str) -> None:
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def save_results_plot_html(rewards):
    # Creating a trace for agent performance
    agent_trace = go.Scatter(x=list(range(1, len(rewards) + 1)), y=rewards, mode='lines', name='Agent Performance',
                             line=dict(color='blue'))

    # Adding traces for human and random performance
    human_trace = go.Scatter(x=[1, len(rewards)], y=[-4000, -4000], mode='lines', name='Human Performance',
                             line=dict(color='green', dash='dash'))
    random_trace = go.Scatter(x=[1, len(rewards)], y=[-17000, -17000], mode='lines', name='Random Performance',
                              line=dict(color='red', dash='dash'))

    # Creating layout
    layout = go.Layout(
        title='Agent Performance Over Episodes',
        xaxis=dict(title='Episodes'),
        yaxis=dict(title='Rewards'),
        showlegend=True
    )

    # Creating figure
    fig = go.Figure(data=[agent_trace, human_trace, random_trace], layout=layout)

    # Saving the plot
    fig.write_html(DQN_PERFORMANCE_PATH)

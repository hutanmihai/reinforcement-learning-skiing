import os
from pathlib import Path

import numpy as np
import cv2 as cv
import plotly.graph_objects as go

from src.constants import PERFORMANCE_PATH_SKELETON


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


def save_results_plot_html(rewards, name_suffix: str):
    agent_trace = go.Scatter(
        x=list(range(1, len(rewards) + 1)), y=rewards, mode="lines", name="Agent Performance", line=dict(color="blue")
    )
    best_player_trace = go.Scatter(
        x=[1, len(rewards)],
        y=[-3272, -3272],
        mode="lines",
        name="Best Player",
        line=dict(color="pink", dash="dash"),
    )
    human_trace = go.Scatter(
        x=[1, len(rewards)],
        y=[-4336.9, -4336.9],
        mode="lines",
        name="Human",
        line=dict(color="green", dash="dash"),
    )
    agent_57_trace = go.Scatter(
        x=[1, len(rewards)],
        y=[-4202.6, -4202.6],
        mode="lines",
        name="Agent 57",
        line=dict(color="purple", dash="dash"),
    )
    random_trace = go.Scatter(
        x=[1, len(rewards)],
        y=[-17098.1, -17098.1],
        mode="lines",
        name="Random",
        line=dict(color="red", dash="dash"),
    )
    noisy_let_duelling_trace = go.Scatter(
        x=[1, len(rewards)],
        y=[-7550, -7550],
        mode="lines",
        name="Noisy LET Duelling",
        line=dict(color="orange", dash="dash"),
    )
    asl_ddqn_trace = go.Scatter(
        x=[1, len(rewards)],
        y=[-8295.4, -8295.4],
        mode="lines",
        name="ASL DDQN",
        line=dict(color="black", dash="dash"),
    )
    advantage_learning_trace = go.Scatter(
        x=[1, len(rewards)],
        y=[-13264.51, -13264.51],
        mode="lines",
        name="Advantage Learning",
        line=dict(color="yellow", dash="dash"),
    )
    rational_dqn_average = go.Scatter(
        x=[1, len(rewards)],
        y=[-23487, -23487],
        mode="lines",
        name="Rational DQN Average",
        line=dict(color="brown", dash="dash"),
    )

    layout = go.Layout(
        title="Agent Performance Over Episodes",
        xaxis=dict(title="Episodes"),
        yaxis=dict(title="Rewards"),
        showlegend=True,
    )

    fig = go.Figure(
        data=[
            agent_trace,
            best_player_trace,
            agent_57_trace,
            human_trace,
            noisy_let_duelling_trace,
            asl_ddqn_trace,
            advantage_learning_trace,
            random_trace,
            rational_dqn_average,
        ],
        layout=layout,
    )

    fig.write_html(PERFORMANCE_PATH_SKELETON + name_suffix + ".html")

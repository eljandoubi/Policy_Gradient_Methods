import logging
import matplotlib
from typing import List, Union

matplotlib.use("agg")
import matplotlib.pyplot as plt


def export_plot(
    ys: List[Union[float, int]], ylabel: str, title: str, filename: str
) -> None:
    """
    Export a plot in filename

    Args:
        ys: (list) of float / int to plot
        filename: (string) directory
    """
    plt.figure()
    plt.plot(range(len(ys)), ys)
    plt.xlabel("Training Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.close()


def get_logger(filename: str) -> logging.Logger:
    """
    Return a logger instance to a file
    """
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    return logger

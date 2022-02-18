import logging
from pathlib import Path

from src.utils.log_color import CustomFormatter

color_console = logging.StreamHandler()
color_console.setLevel(logging.DEBUG)
color_console.setFormatter(CustomFormatter())

# TODO: dl.infer.base fallback 到其它 logger 了
logging.basicConfig(
    format="%(asctime)s [%(levelname)8s] (%(name)s:%(lineno)s) - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG,
    handlers=[color_console]
)

ROOT_DIR = str(Path(__file__).parents[1])

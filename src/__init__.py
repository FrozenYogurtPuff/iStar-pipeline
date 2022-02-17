import logging
from pathlib import Path
logging.getLogger(__name__).addHandler(logging.NullHandler())

logging.basicConfig(
    format="%(asctime)s [%(levelname)8s] (%(name)s:%(lineno)s) - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG
)

ROOT_DIR = str(Path(__file__).parents[1])

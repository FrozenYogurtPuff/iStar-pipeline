import logging
from pathlib import Path

try:
    import colorlog
except ImportError:
    colorlog = None

_format = '%(asctime)s [%(levelname)8s] (%(name)s:%(lineno)s) - %(message)s'
_datefmt = '%Y/%m/%d %H:%M:%S'

console = logging.StreamHandler()
if colorlog:
    formatter = colorlog.ColoredFormatter(
        fmt='%(log_color)s' + _format,
        datefmt=_datefmt,
    )
else:
    formatter = logging.Formatter(
        fmt=_format,
        datefmt=_datefmt
    )
console.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(console)

ROOT_DIR = str(Path(__file__).parents[1])

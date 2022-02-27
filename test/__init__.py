import logging
from typing import Union

try:
    import colorlog
except ImportError:
    colorlog = None  # type: ignore

_format = "%(asctime)s [%(levelname)8s] (%(name)s:%(lineno)s) - %(message)s"
_datefmt = "%Y/%m/%d %H:%M:%S"

console = logging.StreamHandler()
formatter: Union[colorlog.ColoredFormatter, logging.Formatter]

if colorlog:
    formatter = colorlog.ColoredFormatter(
        fmt="%(log_color)s" + _format,
        datefmt=_datefmt,
    )
else:
    formatter = logging.Formatter(fmt=_format, datefmt=_datefmt)
console.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(console)

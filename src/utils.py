import logging
from functools import partial


def setup_logging(debug: bool):
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler = logging.StreamHandler()
    level = logging.DEBUG if debug else logging.INFO
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)

    # Create a partial function for logger.log with the level set
    log_with_level = partial(logger.log, level)
    return log_with_level


import logging
import warnings
warnings.filterwarnings("ignore")

def get_logger(name, log_file_path='./logs/temp.log', logging_level=logging.INFO, 
               log_format='%(asctime)s | %(levelname)s | %(filename)s: %(lineno)s : %(funcName)s() ::\t %(message)s'):
    
    logger = logging.getLogger(name)
    logger.setLevel(logging_level)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d : %(funcName)s() ::\t %(message)s"
    )

    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(logging_level)
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging_level)
    stream_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

def print_log(logger, d: dict, prefix: str = ""):
    parts = []

    for key, value in d.items():
        # Handle Enums like Tasks / Specs
        if not isinstance(key, str):
            key = key.value if hasattr(key, "value") else str(key)

        key = key.replace("_", " ")

        if isinstance(value, float):
            value_str = f"{value:.4f}"
        else:
            value_str = str(value)

        parts.append(f"{key}: {value_str}")

    msg = " | ".join(parts)

    if prefix:
        msg = f"{prefix} | {msg}"

    logger.info(msg)

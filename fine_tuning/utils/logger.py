import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """获取格式化统一的日志记录器。

    Args:
        name (str): 日志记录器名称。

    Returns:
        logging.Logger: 日志记录器实例。
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # 终端输出 Formatter
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # StreamHandler 输出到 stdout
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
    return logger

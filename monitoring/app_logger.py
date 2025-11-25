import logging

def create_logger():
    logger = logging.getLogger(__name__)
    logging.basicConfig(level='INFO')
    return logger
import settings
settings.override_if_main(__name__, 1)

import logging
import os

def create_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(1)
    formatter = logging.Formatter('{asctime} {levelname: <5} {message}', style='{')

    sh = logging.StreamHandler()
    sh.setLevel(settings.get(f'$.logging.level.console.{name}'))
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    if not os.path.isdir('../logs'):
        os.mkdir('../logs')
    fh = logging.FileHandler(f'../logs/{name}.log', 'w', 'utf-8')
    fh.setLevel(settings.get(f'$.logging.level.logfile.{name}'))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

if __name__ == "__main__":
    logger = create_logger('log')
    logger.debug('debug')
    logger.info('info')
    logger.critical('critical')
    logger.error('error')
    try:
        raise Exception('exception raised')
    except:
        logger.exception('exception message')

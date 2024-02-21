import abc
import sys
import logging


def MultipleArgsParser(s):
    return [str(item) for item in s.split(',')]


def InitLogger(name):
    # setup logging
    logger = logging.getLogger(name)
    logger.propagate = False

    # create console handler and set level to info
    stdout = logging.StreamHandler(stream=sys.stdout)
    stdout.setLevel(logging.INFO)

    # create formatter
    formatter = logging.Formatter(
        "%(name)s: %(asctime)s | %(levelname)s | %(message)s")
    stdout.setFormatter(formatter)

    logger.addHandler(stdout)

    # set level to info
    logger.setLevel(logging.INFO)

    return logger


class TrainerMixin(metaclass=abc.ABCMeta):

    def __init__(self) -> None:
        self.logger = InitLogger(__name__)

    @abc.abstractmethod
    def load_data(self):
        pass

    @abc.abstractmethod
    def train(self):
        pass

    def run(self):
        logging.info("Loading data...")
        self.load_data()

        logging.info("Training model...")
        self.train()

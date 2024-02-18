import abc
import logging


def MultipleArgsParser(s):
    return [str(item) for item in s.split(',')]


class TrainerMixin(metaclass=abc.ABCMeta):

    def __init__(self) -> None:
        # setup logging
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

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

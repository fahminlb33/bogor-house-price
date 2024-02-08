import abc
import logging


class ProcessorBase(object, metaclass=abc.ABCMeta):

    def __init__(self) -> None:
        # setup logging
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

    @abc.abstractmethod
    def run(self):
        pass


class TrainerMixin(ProcessorBase, metaclass=abc.ABCMeta):

    def __init__(self) -> None:
        super().__init__()

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


class ETLMixin(ProcessorBase, metaclass=abc.ABCMeta):

    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def extract(self):
        pass

    @abc.abstractmethod
    def transform(self):
        pass

    @abc.abstractmethod
    def load(self):
        pass

    def run(self):
        logging.info("EXTRACT phase")
        self.extract()

        logging.info("TRANSFORM phase")
        self.transform()

        logging.info("LOAD phase")
        self.load()

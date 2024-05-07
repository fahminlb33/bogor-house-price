import abc
import sys
import timeit
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
    self.logger.info("Loading data...")
    start_time = timeit.default_timer()
    self.load_data()
    elapsed = timeit.default_timer() - start_time
    self.logger.info(f"Data loaded in {elapsed:.2f} seconds")

    self.logger.info("Training model...")
    start_time = timeit.default_timer()
    self.train()
    elapsed = timeit.default_timer() - start_time
    self.logger.info(f"Training finished in {elapsed:.2f} seconds")

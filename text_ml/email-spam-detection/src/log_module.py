import logging
import sys

class Logger:
    """
    A base class for other classes to inherit for standardized logging.

    It provides a logger named after the inheriting class, with an option
    to set the logging level during initialization.
    """
    def __init__(self, name=None, log_level=logging.INFO):
        """
        Initializes the Loggable class with a customized logger.

        Args:
            name (str, optional): The name for the logger. If None,
                                  it defaults to the class name.
            log_level (int): The logging level to set for this logger.
                             Defaults to logging.INFO.
        """
        if name is None:
            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            self.logger = logging.getLogger(name)

        # Set the logger's level using the provided argument
        self.logger.setLevel(log_level)

        # Configure a stream handler if one doesn't already exist.
        # This prevents adding multiple handlers when the class is
        # instantiated multiple times.
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
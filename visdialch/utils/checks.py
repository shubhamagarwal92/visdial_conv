"""
Taken from Allennlp.

Functions and exceptions for checking that
the models are configured correctly.
"""
from typing import Union, List

import logging


class ConfigurationError(Exception):
    """
    The exception raised by any AllenNLP object when it's misconfigured
    (e.g. missing properties, invalid properties, unknown properties).

    Use as:
    from visdialch.utils.checks import ConfigurationError
    if cond:
        raise ConfigurationError("Error message")
    """

    def __init__(self, message):
        super(ConfigurationError, self).__init__()
        self.message = message

    def __str__(self):
        return repr(self.message)




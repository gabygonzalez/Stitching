# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import logging

"""
Hopefully this should endup in ros.__doc__
"""

# create logger
# TODO solve multiprocess logger problem(s)...
_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())

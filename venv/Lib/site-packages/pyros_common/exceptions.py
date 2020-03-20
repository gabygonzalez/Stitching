from __future__ import absolute_import

import os



# All Pyros Exception must be pickleable and have a message property
class PyrosException(Exception):
    """Base exception for errors raised by Pyros
       Initialize it with members that can be pickled when moving the exception around
       Redefine the message property to get a meaningful message:

       >>>  # TODO doctest
    """

    def __init__(self, *args):
        """
        Overriding Exception constructor to handle both old and new Exception format...
        :param args:
        :return:
        """
        super(PyrosException, self).__init__(*args)

    # redefining __str__ to get custom message to appear on raise
    def __str__(self):
        return self.message

    @property
    def message(self):
        """
        message as part of core Python Exception has been deprecated.
        We recycle it here for our own use.

        It is not a member, so it will not be serialized and transferred between processes
        Instead it will be reconstructed from members ( we have shared the code for it )
        :return:
        """
        return "An Exception {0} was thrown in Pyros process {1}".format(repr(self), os.getpid())


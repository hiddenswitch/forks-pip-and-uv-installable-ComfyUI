import threading
from contextlib import contextmanager
from contextvars import ContextVar

_interrupt_processing_mutex = threading.RLock()
_interrupt_processing = False
_defer_interrupt_checks = ContextVar("defer_interrupt_checks", default=False)


class InterruptProcessingException(Exception):
    pass


def interrupt_current_processing(value=True):
    global _interrupt_processing
    global _interrupt_processing_mutex
    with _interrupt_processing_mutex:
        _interrupt_processing = value


def processing_interrupted():
    global _interrupt_processing
    global _interrupt_processing_mutex
    with _interrupt_processing_mutex:
        return _interrupt_processing


def throw_exception_if_processing_interrupted():
    global _interrupt_processing
    global _interrupt_processing_mutex
    if _defer_interrupt_checks.get():
        return
    with _interrupt_processing_mutex:
        if _interrupt_processing:
            _interrupt_processing = False
            raise InterruptProcessingException()


@contextmanager
def defer_interruption():
    """Deliver cancellation after a coordinated operation reaches a safe boundary."""
    throw_exception_if_processing_interrupted()
    token = _defer_interrupt_checks.set(True)
    try:
        yield
    finally:
        _defer_interrupt_checks.reset(token)
    throw_exception_if_processing_interrupted()

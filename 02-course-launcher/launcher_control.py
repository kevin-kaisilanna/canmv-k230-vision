"""Cross-firmware loop checkpoint used by the course launcher."""

import os


_return_checker = None


def set_return_checker(checker):
    global _return_checker
    _return_checker = checker


def checkpoint(*args):
    """Check the launcher return key and call firmware exitpoint if present."""
    if not args and _return_checker is not None and _return_checker():
        raise KeyboardInterrupt

    firmware_exitpoint = getattr(os, "exitpoint", None)
    if firmware_exitpoint is not None:
        return firmware_exitpoint(*args)
    return None

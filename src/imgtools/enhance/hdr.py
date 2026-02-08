from ._hdr import __all__  # noqa: F401


def __getattr__(key):
    from . import _hdr

    if (submod := getattr(_hdr, key, None)) is not None:
        return submod
    raise AttributeError(f'module {__name__!r} has no attribute {key!r}')

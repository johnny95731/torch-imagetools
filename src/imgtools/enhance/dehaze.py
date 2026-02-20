from ._dehaze import __all__  # noqa: F401


def __getattr__(key):
    from . import _dehaze

    if (submod := getattr(_dehaze, key, None)) is not None:
        return submod
    raise AttributeError(f'module {__name__!r} has no attribute {key!r}')

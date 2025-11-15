import importlib as _importlib

__version__ = '0.1.0'

_submodules = [  # Lazy loading sub-modules with __getattr__
    'balance',
    'color',
    'enhance',
    'filters',
    'utils',
    'wavelets',
    'statistics',
]
__all__ = _submodules


def __dir__():
    return __all__


def __getattr__(name: str):
    if name in _submodules:
        return _importlib.import_module(f'imgtools.{name}')
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module 'imgtools' has no attribute '{name}'")

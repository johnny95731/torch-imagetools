import os

from tests.basic import path_to_module_path, run_module


def test_all():
    basedir = os.path.dirname(__file__)
    basedir = os.path.relpath(basedir, './')

    entries = os.listdir(basedir)
    entries = [
        os.path.join(basedir, file)
        for file in entries
        if not file.endswith('.pyi')  # `.endswith('.py')` will remove folders
        and file != '__init__.py'
        and file != '__pycache__'
    ]

    basedir = os.path.dirname(__file__)
    for file in entries:
        module_path = path_to_module_path(file)
        if file.endswith('.py'):
            run_module(module_path)


if __name__ == '__main__':
    test_all()

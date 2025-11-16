import importlib
import os
from pathlib import Path


def path_to_module_path(file_path: str) -> str:
    """Convert string from './path/to/file' to 'path.to.file'"""
    rel_path = Path(file_path).relative_to('./')
    if rel_path.suffix == '.py':
        rel_path = rel_path.with_suffix('')
    parts = rel_path.parts
    module_path = '.'.join(parts)
    return module_path


def handle_a_folder(path: str, depth: int):
    entries = os.listdir(path)
    entries = [
        file
        for file in entries
        if not file.endswith('.pyi')  # `.endswith('.py')` will remove folders
        and file != '__init__.py'
        and file != '__pycache__'
    ]

    submodules = []
    submod_attrs: dict[str, list[str]] = {}
    merged_all = []
    print('Start:', path)
    for file in entries:
        full_path = os.path.join(path, file)
        module_path = path_to_module_path(full_path)
        if os.path.isdir(full_path):
            handle_a_folder(full_path, depth + 1)
        elif file.endswith('.py'):
            module = importlib.import_module(module_path)
            root, _ = os.path.splitext(file)
            submodules.append(root)
            _all = getattr(module, '__all__', None)
            if _all is not None:
                _all = sorted(_all)
                submod_attrs[root] = _all
                merged_all += _all

    print('Done:', path)

    if depth > 0:
        submodules = sorted(submodules)
        merged_all = sorted(merged_all)
        # Load docstring of __init.py
        full_path = os.path.join(path, '__init__.py')
        module_path = path_to_module_path(full_path)
        module = importlib.import_module(module_path)
        docstring = module.__doc__

        # Format `__all__ = [...]`
        content = "',\n    '".join(merged_all)
        all_string = f"__all__ = [\n    '{content}',\n]\n"

        # Format `from .{submod} import`
        submodules = sorted(submodules)
        import_string = ''
        for submod in submodules:
            suball = submod_attrs[submod]
            subcontent = ',\n    '.join(suball)
            s = f'from .{submod} import (\n    {subcontent},\n)\n'
            import_string += s

        pycontent = ''
        if docstring is not None:
            pycontent += f'"""{docstring}"""\n\n'
        pycontent += f'{all_string}\n{import_string}'
        with open(
            os.path.join(path, '__init__.py'), 'w', encoding='utf-8'
        ) as f:
            f.write(pycontent)


def main():
    """Generate __init__.pyi files."""
    basedir = './src'
    module_path = os.path.join(basedir, os.listdir(basedir)[0])
    handle_a_folder(module_path, 0)


if __name__ == '__main__':
    main()

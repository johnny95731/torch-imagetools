import os
from importlib.machinery import SourceFileLoader
from pathlib import Path


def path_to_module_path(file_path: str) -> str:
    """Convert string from './path/to/file' to 'path.to.file'"""
    rel_path = Path(file_path).relative_to('./')
    if rel_path.suffix == '.py':
        rel_path = rel_path.with_suffix('')
    parts = rel_path.parts
    module_path = '.'.join(parts)
    return module_path


def handle_a_folder(path: str):
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
            handle_a_folder(full_path)
        elif file.endswith('.py'):
            module = SourceFileLoader(module_path, full_path).load_module()
            root, _ = os.path.splitext(file)
            submodules.append(root)
            _all = getattr(module, '__all__', None)
            if _all is not None:
                _all = sorted(_all)
                submod_attrs[root] = _all
                merged_all += _all

    print('Done:', path)

    merged_all = sorted(merged_all)
    with open(os.path.join(path, '__init__.pyi'), 'w') as f:
        content = "',\n    '".join(merged_all)
        all_str = f"__all__ = [\n    '{content}',\n]\n"
        f.write(all_str)
        f.write('\n')

        # writting `from .{submod} import``
        submodules = sorted(submodules)
        for submod in submodules:
            suball = submod_attrs[submod]
            subcontent = ',\n    '.join(suball)
            s = f'from .{submod} import (\n    {subcontent},\n)\n'
            f.write(s)


def main():
    """Generate __init__.pyi files."""
    basedir = './src'
    module_path = os.path.join(basedir, os.listdir(basedir)[0])
    handle_a_folder(module_path)


if __name__ == '__main__':
    main()

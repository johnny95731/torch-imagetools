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
    if depth > 1:
        return
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
    if depth > 0:
        submodules = sorted(submodules)
        # Load docstring of __init.py
        init_path = os.path.join(path, '__init__.py')
        module_path = path_to_module_path(path).replace('src.', '')
        module = importlib.import_module(path_to_module_path(init_path))
        docstring = module.__doc__

        # Format submod
        _summary = []
        for submod in submodules:
            suball = submod_attrs.get(submod, None)
            if suball is None:
                print(f'"{submod}" has no "__all__".')
                continue
            auto_summary = f"""{'=' * (len(submod))}
{submod.capitalize()}
{'=' * (len(submod))}

.. autosummary::
   :nosignatures:

   {f'\n   '.join(suball)}
"""
            _summary.append(auto_summary)

        # Merge
        title = f':mod:`{module_path}`'
        if docstring is None:
            docstring = ''
        else:
            docstring = f"""{docstring}
---------
"""

        pycontent = f"""{title}
{'=' * len(title)}

.. currentmodule:: {module_path}

{docstring}

Links
-----

{'\n\n'.join(_summary)}
---------


Documents
---------

.. automodule:: {module_path}
   :members:
   :no-docstring:
   :member-order: bysource
"""
        docs_path = f'./docs/source/{"/".join(module_path.split("."))}.rst'
        with open(docs_path, 'w', encoding='utf-8') as f:
            f.write(pycontent)

    print('Done :', path)


def main():
    """Generate .rst files"""
    basedir = './src'
    module_path = os.path.join(basedir, os.listdir(basedir)[0])
    handle_a_folder(module_path, 0)


if __name__ == '__main__':
    main()

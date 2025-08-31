import sys
import os
from pathlib import Path


def find_module_name(file_path: str, project_root: str) -> str:
    rel_path = Path(file_path).relative_to(project_root)
    if rel_path.suffix == '.py':
        rel_path = rel_path.with_suffix('')
    parts = rel_path.parts
    return '.'.join(parts)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: run_module.py <file.py>')
        sys.exit(1)

    file_path = Path(sys.argv[1]).resolve()
    project_root = Path(__file__).parent.resolve()

    module_name = find_module_name(str(file_path), str(project_root))
    print(f'Running as module: {module_name}')

    os.system(f'{os.getcwd()}/.venv/Scripts/python.exe -m {module_name}')

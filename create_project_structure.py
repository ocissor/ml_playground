import os
from pathlib import Path
def create_project_structure(base_path:str):
    dir_names = ['backend', 'frontend', 'data']
    for dir_name in dir_names:
        os.makedirs(base_path / dir_name, exist_ok = True)

    backend_dirs = ['app', 'tests']
    for dir_name in backend_dirs:
        os.makedirs(base_path / 'backend' / dir_name, exist_ok = True)
        with open(base_path / 'backend' / dir_name / '__init__.py', 'w') as f:
            pass

create_project_structure(Path(__file__).parent)
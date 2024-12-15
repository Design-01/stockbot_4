import os
from pathlib import Path

# This will always point to stockbot4 directory
PROJECT_ROOT = Path(os.path.dirname(__file__)).resolve()

def get_project_path(*paths):
    """
    Get absolute path from project root for any file/directory
    """
    return os.path.join(PROJECT_ROOT, *paths)
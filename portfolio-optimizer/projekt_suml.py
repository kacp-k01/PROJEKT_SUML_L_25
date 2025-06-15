import os
import sys

from gui.screen import launch_app

if __name__ == "__main__":
    if not sys.stdout:
        sys.stdout = open(os.devnull, 'w')
    if not sys.stderr:
        sys.stderr = open(os.devnull, 'w')

    launch_app()
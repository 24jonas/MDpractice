import sys

print(sys.path)     # The places this file looks for imports.

# sys.path.append("Path Name")      Adds a path to sys.

# Changing environment variables.
# $ nano ~/.zch_profile
# export PYTHONPATH="(Path Name)"
# $ Python
# $ Import (Module)

# Find the directory of this script.

import os

print(os.getcwd())

# Locate this file
print(os.__file__)

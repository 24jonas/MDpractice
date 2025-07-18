# This file was created to resolve an error encountered in 'TutorialA.py' where GPAW was installed but couldn't find the required PAW datasets.
# The code below verifies that gpaw can find its datasets.

from gpaw import setup_paths
print(setup_paths)

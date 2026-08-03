import glob
import subprocess
import sys
from os.path import abspath, basename, dirname, join

import pytest

cwd = abspath(dirname(__file__))
examples_dir = join(cwd, "..", "examples")


# Discover the examples files by globbing the examples directory
@pytest.fixture(params=glob.glob(f"{examples_dir}/*.py"), ids=lambda x: basename(x))
def py_file(request):
    return abspath(request.param)


def test_demo_runs(py_file):
    subprocess.check_call([sys.executable, py_file])

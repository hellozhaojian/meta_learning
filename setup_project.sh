# mkdir necessary directories
mkdir -p build_tools docs examples test torchtext

# mkfile
touch CODE_OF_CONDUCT.md README.rst readthedocs.yml CONTRIBUTING.md requirements.txt LICENSE codecov.yml \
 pytest.ini setup.py test/__init__.py .travis.yml .gitignore .flake8


export PYTHON_PATH=`pwd`:$PYTHON_PATH

#! /bin/bash

# virtualenv must be installed on your system, install with e.g.
# pip install virtualenv

toy_model_scripts=`dirname "$0"`
base=$toy_model_scripts/../..

mkdir -p $base/venvs

# python3 needs to be installed on your system

python3 -m virtualenv -p python3 $base/venvs/torch3

echo "To activate your environment:"
echo "    source $base/venvs/torch3/bin/activate"

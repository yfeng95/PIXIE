#!/usr/bin/env bash

echo "Creating virtual environment"
python3.7 -m venv pixie-env
echo "Activating virtual environment"

source $PWD/pixie-env/bin/activate
$PWD/pixie-env/bin/pip install -r requirements.txt
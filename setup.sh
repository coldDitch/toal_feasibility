#!/bin/bash
mkdir src/plots
mkdir src/stancodes
mkdir src/res
virtualenv toal_env
source ./toal_env/bin/activate
pip install -r requirements.txt
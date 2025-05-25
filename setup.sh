#!/bin/bash

##
## Simple script to setup the local Python
##
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

pip install numpy torch
pip install gymnasium "gymnasium[mujoco]"
pip install seaborn

echo ""
echo "Run this command to activate Python:"
echo ""
echo "source .venv/bin/activate"




#!/usr/bin/env bash

# download model
python3 scripts/download_model.py

# download pitch and energy labels
gdown 1ir_PwL1BDdvHiwO7uvau7du1kQR7Wb23
tar xvf pitch_energy.tar.gz
mv pitch data/
mv energy data/
rm pitch_energy.tar.gz
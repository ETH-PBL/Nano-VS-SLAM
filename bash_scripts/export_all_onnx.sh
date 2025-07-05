#!/bin/bash

# Directory containing the config files
export_dir="./exported_models"
# List of config names
configs=("S" "S_A" "N" "N_A")

# Loop through all config names
for config_name in "${configs[@]}"; do
    # Construct the config file path
    config_file="--config $config_name"

    # Run the export_onnx.py script for each config
    python3 export_onnx.py $config_file --model_path $export_dir

    python3 export_onnx.py $config_file --model_path $export_dir --model_type KP2DtinyV3
    # Check the exit status of the python script
    if [ $? -ne 0 ]; then
        echo "Error: export_onnx.py failed for config $config_name"
        exit 1
    fi
done

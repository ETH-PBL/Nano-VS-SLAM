#!/bin/bash
root_path="$(pwd)"
datasets_path="$(pwd)/datasets"
hpatches_dir="$datasets_path/HPatches"

mkdir $datasets_path
cd $datasets_path
  
wget http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz
tar -xvf hpatches-sequences-release.tar.gz
mv hpatches-sequences-release $hpatches_dir
rm hpatches-sequences-release.tar.gz
cd $root_path

python -m utils.set_json --key "coco_data_path" --value "$hpatches_dir"

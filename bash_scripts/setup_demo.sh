root_path="$(pwd)"
datasets_path="$(pwd)/datasets"

mkdir checkpoints
cd checkpoints

wget https://polybox.ethz.ch/index.php/s/ZwbDKqrH1Lp5SKW/download
unzip download
rm download

cd datasets_path

mkdir kitti
cd kitti

wget https://polybox.ethz.ch/index.php/s/xhspmIowARAL66T/download
unzip download
rm download

kitti_path = "$(pwd)/kitti06"

cd $root_path
cp ./datasets_template.json ./datasets.json

python -m utils.set_json --key "kitti_path" --value "$kitti_path"
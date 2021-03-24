set -e

source /opt/anaconda3/bin/activate  # Modify this if it does not work
echo y | conda create -n deepfix python=2.7
conda activate deepfix
echo y | conda install pathlib regex subprocess32 tensorflow-gpu==1.0.1 typing

mkdir logs
export PYTHONPATH=.
python data_processing/training_data_generator_cs.py
bash neural_net/1fold-train.sh
python post_processing/proc_cs.py data/Mutation/ > proc_cs.json

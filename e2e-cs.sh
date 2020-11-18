set -e

git clean -dXf

source /opt/anaconda3/etc/profile.d/conda.sh
echo y | conda create -n deepfix python=2.7
conda activate deepfix
echo y | conda install subprocess32 tensorflow-gpu==1.0.1 regex

mkdir logs
export PYTHONPATH=.
python data_processing/training_data_generator_cs.py

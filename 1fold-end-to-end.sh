export PYTHONPATH=.
echo "Step 1 >> Data generation"
bash data_processing/data_generator.sh
echo "Step 2 >> Training"
bash neural_net/1fold-train.sh
echo "Step 3 >> Applying fixes"
bash post_processing/1fold-result_generator.sh

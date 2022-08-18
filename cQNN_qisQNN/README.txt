For cQNN, we used Qiskit version 0.24.1
To set up the cQNN conda environment, use "conda create -n cQNN_env --file cQNN_spec-file.txt"
For qisQNN, we used Qiskit version 0.37.1
To set up the qisQNN conda environment, use "conda create -n qisQNN_env --file qisQNN_spec-file.txt"

For both environments, the specific versions of Qiskit need to be installed.

Minimal examples of how to run the scripts:
1. cQNN:
conda activate cQNN_env
python3 cQNN_HypaCADD.py --cont True --run_name cqnn_genodock --num_cpus 4 --num_layers 3 --layerorder zxz --partition_ratio 6:2:2

2. qisQNN
conda activate qisQNN_env
python3 qisQNN_HypaCADD.py --cont True --run_name qisqnn_genodock --num_cpus 4 --num_layers 3 --partition_ratio 0.33:0.33:0.34

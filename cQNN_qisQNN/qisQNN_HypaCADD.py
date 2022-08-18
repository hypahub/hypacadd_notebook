# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 09:42:44 2021

@author: jacks
"""
# flatfile for the qiskit qnn module as implemented in my qiskit_ml_1 notebook

# %% imports

import numpy as np

from qiskit import Aer, QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit.circuit import ParameterVector
from qiskit.algorithms.optimizers import COBYLA, SPSA, ADAM, TNC

from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

from defs_HypaCADD import *

import time as time

from qiskit_machine_learning.exceptions import QiskitMachineLearningError

import pandas as pd
pd.options.mode.chained_assignment = None #suppresses a warning that shows up in the genodock preprocessing

import os
abspath = os.path.abspath('__file__')
dname = os.path.dirname(abspath)
os.chdir(dname)

import argparse

#for performance metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc

# %% parsing

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Simulate a QNN with the appropriate hyperparameters.")
    parser.add_argument('-e','--epochs', required=False, type=int, help='the desired number of epochs to run', default=10)
    parser.add_argument('-p','--patience', required=False, type=int, help='upper limit for the patience counter used in validation', default=5)
    parser.add_argument('-x','--shots', required=False, type=int, help="the number of shots per circuit simulation", default=100)
    parser.add_argument('-f','--full', required=False, type=bool, help='if set to True, instructs the qnn to run through the entire trainset, regardless of -n hyperparam', default=True)
    parser.add_argument('-i', '--itl_params', required=False, type=float, help='for nonrandom initial parameters, sets every param to this value', default=np.pi/2)
    parser.add_argument('--num_layers', required = False, type=int, help='determines the number of alternating R_ZX and R_XX layers in the QNN', default=6)
    parser.add_argument('--genodock_set', required=False, help='determines which group of features to extract from GenoDock data (1, 2, 3, or 4)', default=4)
    parser.add_argument('-a','--alternate_data', required=False, type=bool, help='if true, feeds data into the net alternating between labels', default=True)
    parser.add_argument('--partition_ratio', required=False, type=str, help="governs the ration of partition sizes in the training, validation, and test sets. a list of the form [train, val, test]", default="1:1:1")
    parser.add_argument('--run_name', required=True, type=str, help='sets the name of the run for reference in the output csv')
    parser.add_argument('--cont', required=False, type=bool, help='determines whether to use the continuous (as opposed to binarized) version of the QNN', default=True)
    parser.add_argument('--shuffle', required=False, type=bool, help='determines whether to shuffle data before alternating', default=False)
    parser.add_argument('--shuffleseed', required=False, type=int, help='a seed for use in shuffling the dataset, if left False and --shuffle=True, will be completely random', default=False)
    parser.add_argument('-o','--optimizer', required=False, type=str, help='determines the Qiskit optimizer used in qnn', default='cobyla')
    parser.add_argument('-c', '--num_cpus', required=False, type=int, help='sets the number of cpus to be used for multiprocessing. vestigial for qisqnn but including so that if the argument is given, wont cause an error', default=4)

    args = parser.parse_args()

    alternate = args.alternate_data
    shuffle=args.shuffle
    parsed_shots=args.shots
    full = args.full
    n_layers = args.num_layers
    shuffleseed = args.shuffleseed
    genodock_set = args.genodock_set
    ratio = args.partition_ratio

    if args.cont:
        converttype = 'cont'
    else:
        converttype = 'bin'

    if args.optimizer.lower() == 'cobyla':
        optimizer = COBYLA
    elif args.optimizer.lower() == 'spsa':
        optimizer = SPSA
    elif args.optimizer.lower() == 'adam':
        optimizer = ADAM
    elif args.optimizer.lower() == 'tnc':
        optimizer = TNC
    else:
        print("problem with parsing optimizer, defaulting to COBYLA")
        optimizer = COBYLA


# %% QNN Parameter and Dataset Initialization


    dataset = import_dataset('genodock_preprocessed','genodock_'+str(genodock_set)+"_norm.txt", shuffle, shuffleseed)
    dataset = get_uq_g(dataset)
    if alternate:
        dataset = alternate_g(dataset)
    else:
        dataset = balance_g(dataset)
    print(f"using dataset of length {len(dataset)}")
    if ratio=='1:1:1':
        ratio = ratio.split(":")
        ratio = [float(entry) for entry in ratio]
        partition_split = (len(dataset)//3)//10*10 #rounds down to nearest 10
    else:
        ratio = ratio.split(":")
        ratio = [float(entry) for entry in ratio]
        partition_split = int(len(dataset)*ratio[0]/sum(ratio))
    print(f'using partition size of {partition_split}')
    train_set, val_set, test_set = train_val_test(dataset, partition_split, ratio)
    train_len = len(train_set)
    get_info_g(train_set, True)
    print("for testing:")
    get_info_g(test_set, True)
    test_len = len(test_set)


    Xtrain, ytrain = convert_for_qiskit(train_set, converttype)
    Xval, yval = convert_for_qiskit(val_set, converttype)
    Xtest, ytest = convert_for_qiskit(test_set, converttype)

# %% circuit def

    n_inputs = len(Xtrain[0])
    n_qubits = n_inputs+1
    n_gates = n_inputs*n_layers
    readout_qubit = n_inputs
    x_params = ParameterVector('x',n_inputs)
    theta_params = ParameterVector('theta', n_gates)

    ##### actual circuit #####
    qc = QuantumCircuit(n_qubits, 1)
    for i in range(n_inputs):
        qc.rx(x_params[i], i)
    qc.barrier()

    param_index=0
    for rep in range(n_layers):

        if rep%2 == 0:
            #for the even numbered layers, build an R_ZX layer
            for q in range(n_inputs):
                qc.rzx(theta_params[param_index], q, readout_qubit)
                param_index +=1
            qc.barrier()

        else:
            #for the odd numbered layers, build an R_XX layer
            for q in range(n_inputs):
                qc.rxx(theta_params[param_index], q, readout_qubit)
                param_index += 1
            qc.barrier()

    #finally, measure
    qc.append(YplusZ_gate, [readout_qubit]) #effectively makes the measurement a y-measurement
    qc.measure(readout_qubit,0)
    qc.append(YplusZ_gate.inverse(), [readout_qubit])

# %% building classifier

    quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=parsed_shots)

    circuit_qnn = CircuitQNN(circuit=qc,
                             input_params=x_params,
                             weight_params=theta_params,
                             interpret=get_readout,
                             output_shape=(2,),
                             quantum_instance=quantum_instance)
    itl_params = np.array([args.itl_params for i in range(n_gates)])
    circuit_classifier = NeuralNetworkClassifier(neural_network=circuit_qnn,
                                             optimizer=optimizer(),
                                             loss= 'absolute_error',
                                             initial_point=itl_params,
                                             warm_start=True)

# %% fitting classifier

    print('training...')
    starttime = time.time()

    # hacking initial parameters
    #circuit_classifier.fit(Xtrain[:1], ytrain[:1]) #have to fit the classifier before i can set the parameters

    #circuit_classifier._fit_result = (np.asarray(itl_params), 0.5, 1)

    # validation
    patience_counter = 0
    best_val_score = 0
    best_params = []
    best_val_epoch = 0

    # fit classifier to data
    for epoch in range(args.epochs):

        circuit_classifier.fit(Xtrain, ytrain)
        this_val_score = circuit_classifier.score(Xval, yval)
        if this_val_score > best_val_score: #validation wrapper
            best_params = circuit_classifier._fit_result.x.tolist()
            best_val_score = this_val_score
            best_val_epoch = epoch
            patience_counter = 0
            print(f"new best validation score {best_val_score}")
        else:
            patience_counter+=1
        if patience_counter == args.patience:
            print("ran out of patience")
            #revert model to best saved parameters
            circuit_classifier._fit_result.x = np.asarray(best_params)
            circuit_classifier._fit_result.fun = best_val_score
            break
    runtime = time.time()-starttime

    # score classifier
    trainscore = circuit_classifier.score(Xtrain, ytrain)
    testscore = circuit_classifier.score(Xtest, ytest)
    print(f'score on train set: {trainscore}')
    print(f'score on test set: {testscore}')
    yscore = circuit_classifier.predict(Xtest)

    fpr1, tpr1, thresholds = roc_curve(ytest, yscore)
    tn, fp, fn, tp = confusion_matrix(ytest, round_yscore(yscore)).ravel()
    AUROC = auc(fpr1, tpr1)
    print(f'  AUC: {AUROC}')
    print(f"  confusion matrix: [tn {tn}, fp {fp}, fn {fn}, tp {tp}]")
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    print(f'  precision = {precision}')
    print(f'  recall = {recall}')

# %% exporting outputs

    # this is gonna be a little janky because i'm going to make it fit in the preexisting spreadsheet
    datadict = {'run_name':["qiskit_"+args.run_name], 'hyperparams':[args],'test_loss':[2*(1-testscore)],'best_val_loss':['trainscore: '+str(trainscore)],'converge_epoch':[str(best_val_epoch)],'AUROC':[AUROC],'confusion_matrix':[(tn,fp,fn,tp)],'precision':[precision],'recall':[recall],'runtime':[runtime],'best_model':[circuit_classifier._fit_result.x.tolist()]}
    df = pd.DataFrame.from_dict(datadict)

    with open('qnn_output.csv', 'a') as file:
        df.to_csv(file, header=False)

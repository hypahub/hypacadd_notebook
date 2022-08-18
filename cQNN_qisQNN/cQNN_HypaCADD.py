# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 10:18:25 2020

@author: jacks
"""
# %% Imports
import time

from numpy import pi
import pandas as pd
pd.options.mode.chained_assignment = None #suppresses a warning that shows up in the genodock preprocessing

# defs contains all of the circuit definitions as well as auxiliary function defs
from defs_HypaCADD import *
import torch
from torch.autograd import Function
import torch.nn as nn
# also the multiprocessing imports
import multiprocessing as mp
from functools import partial

import random

# parsing import
import argparse

#for performance metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc

# setting directory to same directory as this qnn file
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# %% Parsing
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Simulate a QNN with the appropriate hyperparameters.")
    parser.add_argument('-e','--epochs', required=False, type=int, help='the desired number of epochs to run', default=20)
    parser.add_argument('-s','--shift', required=False, type=float, help='the desired shift size for finite difference', default=pi/2)
    parser.add_argument('-x','--shots', required=False, type=int, help="the number of shots per circuit simulation", default=100)
    parser.add_argument('-b','--batch', required=False, type=int, help='batch size for learning loop', default = 20)
    parser.add_argument('-v','--verbose', required=False, type=bool, help='when set to True, prints extra info about the code', default=False)
    parser.add_argument('-n','--num_batches', required=False, type=int, help='total number of batches to train over', default=5)
    parser.add_argument('-l','--learning_rate', required=False, type=float, help='learning rate for the net (fixed)', default=0.1)
    parser.add_argument('-r', '--decay_rate', required=False, type=float, help='decay per batch of the learning rate', default=0.9)
    parser.add_argument('--decay_start', required=False, type=int, help='number of epochs to run before implementing learning rate decay', default=5)
    parser.add_argument('--rand', required=False, type=bool, help='when True, initializes the model with a random set of parameters. By default (=False), initializes model with every parameter as 1.5', default=False)
    parser.add_argument('--eval_set_size', required=False, type=int, help='sets the size of the set used for evaluating the model; data randomly selected from entire evaluation partition', default=100)
    parser.add_argument('--eval_rate', required=False, type=int, help='determines how frequently to evaluate the model, i.e. after every "eval_rate" epochs', default=10)
    parser.add_argument('-c', '--num_cpus', required=False, type=int, help='sets the number of cpus to be used for multiprocessing', default=4)
    parser.add_argument('-i', '--itl_params', required=False, type=float, help='for nonrandom initial parameters, sets every param to this value', default=1.5)
    parser.add_argument('-p','--patience', required=False, type=int, help='upper limit for the patience counter used in validation', default=10)
    parser.add_argument('-f','--full', required=False, type=bool, help='if set to True, instructs the qnn to run through the entire trainset, regardless of -n hyperparam', default=True)
    parser.add_argument('--noise', required=False, type=int, help='percentage noise to be added to the generated toyset', default=20)
    parser.add_argument('--num_layers', required = False, type=int, help='determines the number of alternating R_ZX and R_XX layers in the QNN', default=6)
    parser.add_argument('--genodock_set', required=False, help='determines which group of features to extract from GenoDock data (1, 2, 3, or 4)', default=4)
    parser.add_argument('-a','--alternate_data', required=False, type=bool, help='if true, feeds data into the net alternating between labels', default=True)
    parser.add_argument('--partition_ratio', required=False, type=str, help="governs the ration of partition sizes in the training, validation, and test sets. a list of the form [train, val, test]", default="1:1:1")
    parser.add_argument('--run_name', required=True, type=str, help='sets the name of the run for reference in the output csv')
    parser.add_argument('--cont', required=False, type=bool, help='determines whether to use the continuous (as opposed to binarized) version of the QNN', default=False)
    parser.add_argument('--shuffle', required=False, type=bool, help='determines whether to shuffle data before alternating', default=False)
    parser.add_argument('--seed', required=False, type=int, help='a parameter seed from which to initialize the model', default=False)
    parser.add_argument('--shuffleseed', required=False, type=int, help='a seed for use in shuffling the dataset, if left False and --shuffle=True, will be completely random', default=False)
    parser.add_argument('--layerorder', required=False, type=str, help='order of gate layers in terms of the control qubit (i.e. xzx for a XX, ZX, XX 3-layer net). Defaults to Farhi and Neven alternation between ZX and XX', default='zxzxzx')
    parser.add_argument('--inputpath', required=False, type=str, help='Path to the new dataset', default='')

    args = parser.parse_args()

    #indexing several of the arguments to simpler variables for easier access
    alternate = args.alternate_data
    shuffle = args.shuffle
    parsed_shift = args.shift
    parsed_shots = args.shots
    parsed_batch_size = args.batch
    parsed_dataset = 'genodock'
    verbose = args.verbose
    parsed_num_batches = args.num_batches
    parsed_epochs = args.epochs
    parsed_learning_rate = args.learning_rate
    parsed_decay_rate = args.decay_rate
    parsed_decay_start = args.decay_start
    parsed_rand = args.rand
    eval_set_size = args.eval_set_size
    eval_rate = args.eval_rate
    num_cpus = args.num_cpus
    itl_params = args.itl_params
    patience=args.patience
    full = args.full
    noise = args.noise
    num_layers = args.num_layers
    shuffleseed = args.shuffleseed

    genodock_set = args.genodock_set
    ratio = args.partition_ratio

    seed = args.seed
    inputpath = args.inputpath

    subset = genodock_set


    # %% Data preprocessing


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
    traindataset = torch.utils.data.DataLoader(train_set, batch_size = parsed_batch_size, shuffle=False)
    print("for the validation:")
    get_info_g(val_set, True)
    print("for testing:")
    get_info_g(test_set, True)
    val_set = torch.utils.data.DataLoader(val_set, shuffle=False)
    test_len = len(test_set)
    test_set = torch.utils.data.DataLoader(test_set, shuffle=False)




    #for use in automatically determining the appropriate circuit size
    n_qubits = get_n_qubits(test_set)
    num_params = (n_qubits-1)*num_layers

    #generating the starting parameters
    if parsed_rand:
        rand_params = [random.uniform(0,pi) for i in range(num_params)]
        params = rand_params
        print("starting params:", rand_params)
    else:
        starting_params = [itl_params for i in range(num_params)]
        params = starting_params

    print(f'Training cQNN on {parsed_dataset} {subset} with a partition size of {len(dataset)}')

    # %% Training Loop

    starttime = time.time() #for use in calculating total runtime

    #initializing variables for the training loop
    learning_rate = parsed_learning_rate
    model = Net(parsed_shots, parsed_shift, n_qubits, num_layers, args.layerorder)
    model.train()
    print("Training network complete")
    batches_loss_list = []
    eval_avgs = []
    batch_size = parsed_batch_size
    if full:
        num_batches = train_len//batch_size #automatically chooses the maxium possible number of batches
    else:
        num_batches = parsed_num_batches #caps the number of batches to process, i.e. the loop will cycle through the first batch_size batches of the dataset
    counter = 0
    epoch_counter = 0
    eval_counter = 0
    epochs = parsed_epochs

    #initializing variables for the validation loop
    patience_counter = 0
    out_of_patience = False
    best_val_error = 2 #just setting this as something arbitrarily large so that the very first validation will be the new best validation error
    best_params = []
    best_val_epoch = 0

    for i in range(epochs):
        batch_counter = 0
        if verbose:
            print(f"Epoch params = {params}")
        for batch in traindataset:
            if batch_counter == num_batches:
                break #break the loop if max num batches reached
            print(f'*** working on batch number {batch_counter}, which is overall calculation {num_batches*epoch_counter+batch_counter} ***')
            items, labels = batch

            batchset = list(zip(items, labels))
            pool = mp.Pool(processes=num_cpus)
            fix_forward=partial(model.forward,thetas = params, continuous = bool(args.cont)) #for multiprocessing. model.forward will return (prediction, gradient, loss) for each datapoint
            batch_results = pool.map(fix_forward,batchset)
            pool.close()
            pool.join()
            this_loss_list = []
            this_grad_list = []
            this_output_list = []
            # since the return of the batch_forward is a list of tuples, manually append each tuple element to the appropriate
            # batch list (in other words rearranging so that all outputs are in a list, all gradients are in a list, and all losses are in a list)
            for item in batch_results:
                this_output_list.append(item[0])
                this_grad_list.append(item[1])
                this_loss_list.append(item[2])
            avgbatchloss = sum(this_loss_list)/len(this_loss_list)
            batches_loss_list.append(avgbatchloss) #keeping track of how batch losses evolve as the dataset is processed
            batchgrad = sum(this_grad_list)/len(this_grad_list) #calculating the average gradient for the batch
            params = update_params(params, learning_rate, batchgrad, avgbatchloss) #updates parameters using gradient descent
            if full: #if running throuh the entire datset (default), validates at regular intervals instead of between epochs
                if batch_counter%eval_rate==0:
                    model.eval()
                    eval_avg, eval_losses = evaluate(model, params, eval_set_size, num_cpus, args.cont, val_set)
                    eval_avgs.append(eval_avg)
                    if eval_avg < best_val_error:
                        best_params = params #catalog the best performing model
                        best_val_error = eval_avg
                        print("new best validation error:", str(best_val_error))
                        patience_counter = 0 #resets patience counter
                        best_val_epoch = epoch_counter
                    else:
                        patience_counter +=1 #increases patience counter every time the model does not improve
                    model.train()
                    if patience_counter==patience:
                        out_of_patience = True
                        print("ran out of patience")
                        break #breaks out of training loop... out_of_patience flag prevents the epoch validation after
            batch_counter+=1

        if parsed_decay_start != False:
            if epoch_counter > parsed_decay_start: #waits until the appropriate epoch to start the learning rate decay
                learning_rate = learning_rate*parsed_decay_rate # decreases the learning rate according to the appropriate hyperparameter

        #validation for this epoch
        if not out_of_patience:
            model.eval()
            eval_avg, eval_losses = evaluate(model, params, eval_set_size, num_cpus, args.cont, val_set)
            eval_avgs.append(eval_avg)
            if eval_avg < best_val_error:
                best_params = params #catalog best model
                best_val_error = eval_avg
                print("new best validation error:", str(best_val_error))
                best_val_epoch = epoch_counter
                patience_counter = 0
            else:
                patience_counter +=1
            model.train()
            if patience_counter==patience:
                print("ran out of patience")
                break
        if out_of_patience:
            break

        epoch_counter +=1
    runtime = time.time()-starttime
    print()
    print("*** Final stats ***")
    print("   losses:", batches_loss_list)
    print("   runtime:", runtime)
    print("   best model:", best_params)
    print("   eval avgs:", eval_avgs)
    test_loss, test_losses = evaluate(model, best_params, test_len, num_cpus, args.cont, test_set)
    print("   loss on testset with best parameters:", test_loss)
    print(f"   converged after {best_val_epoch+1} epochs")
    print()
    print("*** Metrics ***")
    ytrue, yscore = get_ytrue_yscore(model, best_params, test_set)
    fpr1, tpr1, thresholds = roc_curve(ytrue, yscore)
    tn, fp, fn, tp = confusion_matrix(ytrue, round_yscore(yscore)).ravel()
    AUROC = auc(fpr1, tpr1)
    print(f'  AUC: {AUROC}')
    print(f"  confusion matrix: [tn {tn}, fp {fp}, fn {fn}, tp {tp}]")
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    print(f'  precision = {precision}')
    print(f'  recall = {recall}')

    #sending output to a csv
    datadict = {'run_name':[args.run_name], 'hyperparams':[args],'test_loss':[test_loss],'best_val_loss':[best_val_error],'converge_epoch':[best_val_epoch+1],'AUROC':[AUROC],'confusion_matrix':[(tn,fp,fn,tp)],'precision':[precision],'recall':[recall],'runtime':[runtime],'best_model':[best_params]}
    df = pd.DataFrame.from_dict(datadict)
    with open('qnn_output.csv', 'a') as file:
        df.to_csv(file, header=False)

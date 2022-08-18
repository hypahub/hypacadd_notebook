
#%% Imports

import torch
from torch.autograd import Function
import torch.nn as nn

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.extensions import UnitaryGate
import numpy as np
from numpy import pi
import collections
import random
import pandas as pd
import multiprocessing as mp
from functools import partial
import matplotlib
from sklearn.metrics import auc, roc_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

#%% YplusZ_gate definition (for use in y-measurement)

yz_mtx = np.zeros((2,2), dtype=complex)
yz_mtx[0][0] = 1
yz_mtx[0][1] = -1j
yz_mtx[1][0] = 1j
yz_mtx[1][1] = -1
yz_mtx = yz_mtx*(1/np.sqrt(2)) #making sure the matrix is unitary
YplusZ_gate = UnitaryGate(yz_mtx, label="Y+Z") #casts the matrix as a qiskit circuit gate

#%% QNN class definitions

# these largely follow the qiskit textbook hybrid quantum-classical neural network implementation
#  https://qiskit.org/textbook/ch-machine-learning/machine-learning-qiskit-pytorch.html

class QCircuit:
    """Defining an interface for easy circuit interaction in the network. Implements the Farhi and Neven architecture with the ording of layers governed by layerorder hyperparam."""

    def __init__(self, n_qubits, n_layers, backend, shots, layerorder):

        # --- Circuit definition ---
        self._circuit = QuantumCircuit(n_qubits, 1)

        self.n_params = (n_qubits-1)*n_layers
        input_qubits = [i for i in range(n_qubits-1)] #all but the last qubit in the circuit are input qubits
        readout_qubit = [n_qubits-1] #the last qubit in the circuit is the readout

        #defining the quantum circtuit Parameters in a list
        self.thetalist=[]
        for i in range(self.n_params):
            name="theta"+str(i)
            self.thetalist.append(Parameter(name)) #this is a qiskit Parameter

        self._circuit.x(readout_qubit) #initialize readout to |1>
        self._circuit.barrier() #for clarity in viewing circuit

        #building the layers
        param_index = 0
        for rep in range(n_layers):
            gate = layerorder[rep]
            if gate == 'z':
                #if the next character in layerorder hyperparam is z, add an RZX gate
                for q in input_qubits:
                    self._circuit.rzx(self.thetalist[param_index], q, readout_qubit)
                    param_index +=1
                self._circuit.barrier()

            elif gate == 'x':
                #if the next character in layerorder hyperparam is x, add an RXX gate
                for q in input_qubits:
                    self._circuit.rxx(self.thetalist[param_index], q, readout_qubit)
                    param_index += 1
                self._circuit.barrier()

            else:
                print("error with --layerorder hyperparam")

        #finally, measure
        self._circuit.append(YplusZ_gate, [readout_qubit]) #effectively makes the measurement a y-measurement
        self._circuit.measure(readout_qubit,0)
        self._circuit.append(YplusZ_gate.inverse(), [readout_qubit]) #since this gate is after measurement, it's unnecessary, but i'm including it for completeness

        # ---------------------------

        self.backend = backend
        self.shots = shots

    def run(self, itlzd_circuit, thetas):
        """
        This layer expects a quantum circuit initialized to represent a binary data input. Thetas is the parameter list
        which is continuously tweaked by the net. Returns expectation value of y-measurement on the readout qubit.
        """

        #first, bind the circuit Parameters to run the circuit
        binds = {}
        for i in range(len(thetas)):
            param = self.thetalist[i]
            binds[param] = thetas[i]

        #then take the input circuit and merge it with the layered circuit
        self.combinedcircuit = itlzd_circuit.compose(self._circuit)

        #and finally run the circuit
        job = execute(self.combinedcircuit,
                      self.backend,
                      shots = self.shots,
                      parameter_binds = [binds])
        result = job.result().get_counts()

        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)

        probabilities = counts / self.shots

        expectation = np.sum(states*probabilities)

        return np.array([expectation])


#and defining the layer functions for use in the net
#I'm trying to be consistent with calling the dataset input to the net 'data' and the initialized circuit input to the QuantumLayer 'input'
class QFunction(Function):
    """Function definitions for the quantum circuit layer."""

    @staticmethod
    def forward(ctx, input, quantum_circuit, shift, thetas):
        """Forward pass computation"""
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit


        expectation_z = ctx.quantum_circuit.run(input, thetas)
        result = torch.tensor(np.array([expectation_z]))

        return result

    #the textbook then defines the backward pass here, but instead I'll define it as an external function
    #I'm doing this because I couldn't get the .backward() method to work since I'm defining a custom loss function

#the vast majority of these class definitions follow the Qiskit textbook closely. small adjustments have been made to fit our purposes, but they should be fairly clear
class QLayer(nn.Module):
    """Quantum layer definition"""

    def __init__(self, backend, shots, shift, n_qubits, n_layers, layerorder):
        super(QLayer, self).__init__()
        self.quantum_circuit = QCircuit(n_qubits, n_layers, backend, shots, layerorder)
        self.shift = shift

    def forward(self, input, thetas):
        self.thetas = thetas
        return QFunction.apply(input, self.quantum_circuit, self.shift, thetas)

class Net(nn.Module):
    def __init__(self, shots, shift, n_qubits, n_layers, layerorder, backend='qasm_simulator'):
        super(Net, self).__init__()
        self.quantum_layer = QLayer(Aer.get_backend(backend), shots, shift, n_qubits, n_layers, layerorder)
        fig = self.quantum_layer.quantum_circuit._circuit.draw(output='mpl',plot_barriers=False, fold = 50)
        fig.savefig("QNN_3-layer.eps",bbox_inches='tight',dpi=300)

        self.n_qubits = n_qubits
        self.n_layers = n_layers

    def forward(self, data, thetas, continuous):
        if continuous:

            itlzd_circuit = initialize_coarse(data, self.n_qubits)
        else:

            itlzd_circuit = initialize(data)


        output = self.quantum_layer(itlzd_circuit, thetas) #returns the expectation value of the qnn, i.e. its prediction
        expectation_grad = get_expectation_grad(data, thetas, self.quantum_layer, itlzd_circuit, self.quantum_layer.shift)
        grad = loss_grad(data, expectation_grad)
        loss = loss_function(data,output)
        return (output, grad, loss)

    def eval_forward(self, thetas, continuous, data):
        if continuous:
            itlzd_circuit = initialize_coarse(data, self.n_qubits)
        else:
            itlzd_circuit = initialize(data)

        output = self.quantum_layer(itlzd_circuit, thetas)
        loss = loss_function(data, output)
        return loss

    def roc_forward(self, thetas, continuous, data):
        if continuous:
            itlzd_circuit = initialize_coarse(data, self.n_qubits)
        else:
            itlzd_circuit = initialize(data)
        output = self.quantum_layer(itlzd_circuit, thetas)
        return output.item()

# %% Net_cross and associated defs

#below are some slightly different class definitions for the QNN architecture that uses cross gates
#aside from the circuit, most of the code is the same as that of the original QCircuit class

#%% General function definitions

# -- for data preprocessing --
def get_n_qubits(dataset):
    """Extracts the number of qubits needed to learn on a given dataset"""
    n_qubits = 0
    for data in dataset:
        input, label = data
        n_qubits = len(input[0])+1
        break
    return n_qubits

def initialize(data, THRESHOLD=0.5):
    """initializes a quantum circuit for tensor of arbitrary length. Expects that data is a tensor of the form (input, label)"""
    binarized = (data[0]>THRESHOLD)
    binarized = binarized.int()
    flattened = binarized.flatten().tolist()
    qc = QuantumCircuit(len(flattened)+1, 1)
    for q in range(len(flattened)):
        if flattened[q]==1:
            qc.x(q)
    return qc

def remap(expectation):
    """Remaps expectation values in the [0, 1] range to the [-1, 1] range to make them compatible with the loss function defined in Farhi and Neven 2018"""
    return 1-2*expectation

def loss_function(data, output):
    """Takes in data and compares net output to target output, returning the loss between the two."""
    #data will be a tuple of input, label
    input, label = data

    #first, remap the labels
    # 3 and 6 are from testing on MNIST data
    # if label == 3:
    #     label = 1
    if label == 1:
        label = -1
    # elif label == 6:
    #     label = -1
    elif label == 0:
        label = 1
    else:
        return("error with data label")

    expectation = output[0].item() #the .item() method returns the actual numerical value stored in the expectation tensor
    remapped = remap(expectation) #remapping this to the [-1,1] range

    #then use the loss definition in the Farhi Neven paper (Equation 6)
    return 1 - label*remapped

def rescale(dataset):
    """
    Rescales a dataset such that each input's max value is 1, min value is 0, and all other values are adjusted accordingly.
    Meant to be used before binarize(). Used with MNIST data.
    Expects dataset with only positive values. Datasets with negative values should first be shifted so that the min val is zero.
    """
    newdataset = []
    for data, label in dataset:
        max = torch.max(data)
        scaled = data/max
        newdataset.append((scaled, label))
    return newdataset

def binarize(data, THRESHOLD=0.5):
    """Binarizes a data input before it is fed into the QNN. Expects data in the form (input, label)"""
    input, label = data
    binarized = (input>THRESHOLD).float()
    return (binarized, label)

def binarize_dataset(dataset, THRESHOLD=0.5):
    """Applies binarization to an entire dataset. Like binarize(), expects data in the form (input, label)"""
    binarized_dataset=[]
    for data in dataset:
        binarized_dataset.append(binarize(data, THRESHOLD))
    return binarized_dataset

def coarsen(data):
    input, label = data
    coarse = torch.tensor([round(item.item(),1) for item in input.flatten()])
    return (coarse, label)

def coarsen_dataset(dataset):
    new = []
    for data in dataset:
        new.append(coarsen(data))
    return new

def initialize_coarse(data, n_qubits='error'): #issue w the n_qubits here
    """given an input which is a 4x4 pixel intensity array from MNIST, initializes the corresponding quantum circuit"""
    if n_qubits == 'error':
        return print("Requires n_qubits argument")
    else:
        qc = QuantumCircuit(n_qubits, 1)
        coarse, label = coarsen(data)
        flattened = coarse.flatten().tolist()
        for q in range(len(flattened)):
            theta = pi*flattened[q]
            qc.rx(theta, q)
        return qc

def evaluate(model, params, eval_set_size,num_cpus, continuous, dataloader): #formerly mp_evaluate
    counter = 0
    evalset = []
    for data in dataloader:
        if counter == eval_set_size:
            break
        evalset.append(data)
        counter +=1
    pool = mp.Pool(processes=num_cpus)
    mp_eval_forward = partial(model.eval_forward, params, continuous)
    eval_losses = pool.map(mp_eval_forward,evalset)
    pool.close()
    pool.join()
    eval_avg = sum(eval_losses)/len(eval_losses)
    return (eval_avg, eval_losses)

# -- for running the qnn --

def get_expectation_grad(data, params, quantum_layer, itlzd_circuit, shift):
    """returns the gradient of the expectation value for the quantum circuit, calculated through finite difference"""
    input, label = data
    params_list = np.array(params)

    # create a list off all parameters shifted appropriately to pull from in the calculation
    shift_right = params_list + np.ones(params_list.shape) * shift
    shift_left = params_list - np.ones(params_list.shape) * shift

    gradients = []
    for i in range(len(params_list)):
        # take the original parameters but replace parameter i with the appropriate shifted value
        shifted_params_right = params_list.copy()
        shifted_params_right[i] = shift_right[i]

        shifted_params_left = params_list.copy()
        shifted_params_left[i] = shift_left[i]

        #calculate the expectation value for each shift
        expectation_right = quantum_layer(itlzd_circuit, shifted_params_right)
        expectation_left = quantum_layer(itlzd_circuit, shifted_params_left)

        #remap to the [-1,1] range
        remapped_right = remap(expectation_right.item())
        remapped_left = remap(expectation_left.item())

        # calculate the gradient
        gradient = (remapped_right - remapped_left)/(2*shift)
        gradients.append(gradient)

    return torch.tensor([gradients])

def loss_grad(data, expectation_grad):
    """using the gradient of the expectation, calculates the gradient of the loss. specific to mnist"""
    input, label = data
    # 3 and 6 labels are leftovers from working with MNIST data
    # if label == 3:
    #     label = 1
    if label == 1:
        label = -1
    # elif label == 6:
    #     label = -1
    elif label == 0:
        label = 1
    else:
        return print("label error")
    #since loss = 1 - l(z)*expectation, gradient of loss should be = -l(z)*(gradient of expectation)
    return - label * expectation_grad

def update_params(params, learning_rate, gradient, loss):
    """Given a parameter set and a gradient, updates the parameters accordingly."""
    updated_params = []
    grad_norm = torch.norm(gradient).item()
    for i in range(len(params)):
        param = params[i]

        #this parameter update follows Equation 32 of the Farhi and Neven paper
        param -= learning_rate * loss * gradient[0][i] / (grad_norm**2)

        updated_params.append(param.item())
    return updated_params

#%% Function definitions specific to the MNIST problem

def get_uq_g(dataset):
    """Removes all duplicate and conflicting inputs from a dataset."""
    uq = []
    num_0 = 0
    num_1 = 0
    num_overlap = 0
    dict = collections.defaultdict(set)
    for data, label in dataset:
        key = tuple(data.flatten().tolist())
        dict[key].add(label)
    for item in dataset:
        data, label = item
        key = tuple(data.flatten().tolist())
        if dict[key] == {0}:
            num_0 +=1
            uq.append(item)
        elif dict[key] == {1}:
            num_1 +=1
            uq.append(item)
        elif dict[key] == {0,1}:
            num_overlap +=1
        else:
            print("Error with item", item)
            break
    return uq


def array_to_dataset(array):
    """Converts data arrays into appropriate form for use in QNN. Assumes each entry in array is a seperate datapoint w the last column corresponding to label."""
    dataset = []
    for entry in array:
        input = torch.tensor(entry[:len(entry)-1])
        label = entry[len(entry)-1]
        dataset.append((input,label))
    return dataset

def import_dataset(dirname, filename, shuffle=False, shuffleseed=False):
    """
    Imports appropriately-formatted text matrix, converting to array then to dataset.
    Includes options to shuffle randomly or according to a given seed.
    """
    array = np.loadtxt(dirname+"/"+filename)
    if shuffle:
        if shuffleseed==False:
            np.random.shuffle(array)
        else:
            np.random.seed(shuffleseed)
            np.random.shuffle(array)
    return array_to_dataset(array)

def train_val_test(dataset, scale=100, ratio=[1,1,1]):
    """splits dataset into training, validation, and test partitions according to list 'ratio' scaled by 'scale'"""
    scale = int(scale) #no idea why this needs to be here because scale should already be an int, but for some reason it's a string

    part1 = scale
    part2 = part1 + int(len(dataset)*ratio[1]/sum(ratio))
    part3 = part2 + int(len(dataset)*ratio[2]/sum(ratio))
    train = dataset[:part1]
    val = dataset[part1:part2]
    test = dataset[part2:part3]
    return (train, val, test)

def get_info_g(dataset, verbose=False):
    """Determines the number of inputs labeled one and zero in a dataset."""
    zeros = 0
    ones = 0
    for data in dataset:
        input, label = data
        if label == 0:
            zeros+=1
        elif label ==1:
            ones+=1
    if verbose:
        print(f'In this dataset, there are {zeros} inputs labeled "0" and {ones} inputs labeled "1".')
    return (ones, zeros)

# note that this is overruled by the alternate_g function, which is used by default
def balance_g(dataset):
    ones, zeros = get_info_g(dataset)
    ratio = (zeros-ones)/(zeros)
    balanced = []
    for item in dataset:
        data, label = item
        if label == 0:
            if random.random()>ratio:
                balanced.append(item)
        else:
            balanced.append(item)
    return balanced

def alternate_g(dataset):
    ones, zeros = sort_dataset(dataset)
    return coallated_dataset(ones, zeros)

#note sort_df below is more general. keeping both for convenience
def sort_genodock(df):
    "used in data preprocessing"
    df1 = df[df['BA_Change_Vina']==0]
    df2 = df[df['BA_Change_Vina']==1]
    return (df1, df2)

def sort_df(df, labelcolname):
    "used in data preprocessing. splits dataset according to label"
    df1 = df[df[labelcolname]==0]
    df2 = df[df[labelcolname]==1]
    return (df1,df2)

def alternate_df(df1, df2):
    if len(df1) < len(df2):
        length = len(df1)
        new_df = df1.copy()
    else:
        length = len(df2)
        new_df = df2.copy()
    for i in range(length):
        new_df.loc[i] = df1.iloc[i] #changing to iloc from loc
        new_df.loc[i+1]= df2.iloc[i] #changing to iloc from loc

    return new_df

def sort_dataset(dataset, a_label=1):
    labeled_a = []
    labeled_b = []
    for data in dataset:
        input, label = data
        if label == a_label:
            labeled_a.append(data)
        else:
            labeled_b.append(data)
    return (labeled_a, labeled_b)

def coallated_dataset(set1, set2):
    dataset = []
    if len(set1)<len(set2):
        length = len(set1)
    else:
        length = len(set2)
    for i in range(length):
        dataset.append(set1[i])
        dataset.append(set2[i])
    return dataset


def get_ytrue_yscore(model, params, dataset, binarize=False, cont=True):
    ytrue = []
    yscore = []
    for data in dataset:
        input, label = data
        ytrue.append(label.item())
        if binarize:
            yscore.append(round(model.roc_forward(params,cont, data)))
        else:
            yscore.append(model.roc_forward(params, cont, data))
    ytrue = np.asarray(ytrue)
    yscore = np.asarray(yscore)
    return (ytrue,yscore)

def round_yscore(yscore):
    rounded = []
    for item in yscore:
        rounded.append(round(item))
    return rounded

# %% Functions useful for logistic regression:

def ds_to_df(ds):
    index = 0
    for item in ds:
        input, label = item
        input = input.tolist()
        label = label.item()
        input.append(label)
        if index==0:
            df = pd.DataFrame([input], columns=[str(n) for n in range(len(input))])
        else:
            df.loc[index] = input
        index +=1
    return df

def convert_row(row):
    vals = row.values
    input = vals[:len(vals)-1]
    label = vals[len(vals)-1]
    return (torch.tensor(input), label)

def df_to_dataset(df):
    dataset = []
    for i in range(len(df)):
        row = df.iloc[i]
        dataset.append(convert_row(row))
    return dataset

def coarsen_df(df):
    df= df/df.abs().max()
    return df.round(1)

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def threshold_col(df, col_name):
    """thresholds the column based on the median value"""
    threshold = df[col_name].median()
    boolean = df[col_name]>threshold
    return boolean

def binarize_df(df):
    new_df = pd.DataFrame()
    for col_name in df.columns:
        if col_name in ["bind_site","PPH_consequence","SIFT_consequence"]:
            new_df = pd.concat([new_df,df[col_name]], axis=1)
        else:
            new_df = pd.concat([new_df,threshold_col(df, col_name)], axis=1)
    return new_df.astype(float)

# %% defs for qisQNN

def get_readout(x):
    """
    given an output string, returns only the readout qubit. Since qiskit orders qubits in reverse order of importance,
    the readout qubit will be the first number of the output string.
    """
    return(int(str(x)[0]))

def convert_for_qiskit(dataset, datatype="cont"):
    X = []
    y = []
    for input, label in dataset:
        input = input.numpy()
        if datatype == 'cont':
            input = input.round(1)
        elif datatype == 'bin':
            input = input.round(0)
        else:
            return(print('set datatype to cont or bin'))
        X.append(input*np.pi)
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    return(X,y)

def remove_feature(featnum, num_layers, params):
    """
    writing a function to adjust models for smaller feature sets by removing all parameters corresponding to a given feature.
    features are numbered starting from 0, and whichever number is given to the function will be removed from the parameter set.
    """
    num_features = len(params)/num_layers
    newparams = []
    for i in range(len(params)):
        if i%num_features == featnum:
            pass
        else:
            newparams.append(params[i])
    return newparams

# %% import data def
def import_data(parsed_dataset, parsed_batch_size, partition_size, genomics_dataset, genodock_set, alternate, continuous, shuffle, converttype, ratio = [1,1,1]):

    if parsed_dataset=="genodock":
        dataset = import_dataset('genodock_preprocessed','genodock_'+str(genodock_set)+".txt", shuffle)
        if alternate:
            dataset = alternate_g(dataset)
        train_set, val_set, test_set = train_val_test(dataset, partition_size, ratio)
        Xtrain, ytrain = convert_for_qiskit(train_set, converttype)
        Xval, yval = convert_for_qiskit(val_set, converttype)
        Xtest, ytest = convert_for_qiskit(test_set, converttype)
        get_info_g(train_set, True)
        print("for the validation:")
        get_info_g(val_set, True)
        print("for testing:")
        get_info_g(test_set, True)
        test_set = torch.utils.data.DataLoader(test_set, shuffle=False)

    else:
        print("error with dataset hyperparameter!")
    return (train_set,val_set,test_set,Xtrain,ytrain,Xval,yval,Xtest,ytest)

#%% ROC and saliency

def get_roc_df(models, test_set, Xtest, ytest,parsed_shots, parsed_shift, n_qubits, num_layers=3, full_cross=False, layerorder='zxzxzx'):
    """
    Expects a list of models, each of which is a touple of the form
    (params [list if qnn, full model if nn], model_name [str], modeltype ['nn' or 'qnn']).
    Returns an AUC list and a DataFrame which can be use to plot ROC curves.
    """
    namelist = []
    fprlist = []
    tprlist = []
    AUClist = []

    for params, name, modeltype in models:
        if modeltype == 'qnn':
            if '6' in name:
                num_layers = 6
            model = Net(parsed_shots, parsed_shift, n_qubits, num_layers, layerorder)
            model.eval()
            ytrue, yscore = get_ytrue_yscore(model, params, test_set)
            if 'qisQNN' in name: # prediction flip hack that I still need to get to the root of !!
                yscore = 1-yscore

        elif modeltype.lower() == 'crossqnn':
            if "full" in name.lower():
                full_cross = True
            model = Net_cross(parsed_shots, parsed_shift, n_qubits, num_layers, full_cross)
            model.eval()
            ytrue, yscore = get_ytrue_yscore(model, params, test_set)

        elif modeltype == 'logreg':
            yscore = params.decision_function(Xtest)
            ytrue = ytest
        else:
            yscore = [item[1] for item in params.predict_proba(Xtest)]
            ytrue = ytest

        print(f'for {name}, the confusion matrix is: {confusion_matrix(ytrue, round_yscore(yscore)).ravel()}')
        fpr1, tpr1, thresholds = roc_curve(ytrue, yscore)
        fprlist += list(fpr1)
        tprlist += list(tpr1)

        auc1 = auc(fpr1, tpr1)
        AUClist.append((auc1, name))
        namelist += [name+f' (AUC = {round(auc1,2)})']*len(fpr1)

    roc_dict = {"model_name":namelist,
               "fpr":fprlist,
               "tpr":tprlist}

    return (AUClist, pd.DataFrame.from_dict(roc_dict))

def get_shift_loss(data, model, params, shift, modeltype, continuous=True):
    """returns a list of the unshifted loss followed the loss when each element is shifted independently"""
    input, label = data
    losses = []
    losses.append(model.eval_forward(params, continuous, data))
    inputlist = input[0].tolist()
    for i in range(len(inputlist)):
        shiftedlist = inputlist.copy()
        shiftedlist[i] -= shift
        shiftedinput = torch.Tensor(shiftedlist)
        shifteddata = (shiftedinput, label)
        shift_loss = model.eval_forward(params, continuous, shifteddata)
        losses.append(shift_loss)
    return losses



def get_saliency(dataset, n_qubits, modelset, shift, maxdata, parsed_shots, parsed_shift, num_layers=3, full_cross=False, layerorder='zxzxzx', signed=False, convert=True):
    """
    Returns a dataframe which describes input saliencies across various (trained) QNN models. Arguments are as follows:
        'dataset': The data for which saliency is desired. Data is expected in the form (input, label), and dataset is expected to be a list of data tuples.
        'n_qubits': Number of qubits in the QNN, i.e. number of input values + 1
        'modelset': A list of final models for which the saliency is wanted. Each entry is a tuple with (model, modeltype, thetalist)
        'shift': Amount by which parameters should be shifted.
        'maxdata': Caps the number of datapoints for which the saliency is computed
    """
    columns = []
    columns.append('datanum')
    columns.append("modelname")
    for n in range(n_qubits-1):
        columns.append("feature_"+str(n))
    df = pd.DataFrame(columns = columns)
    modelcounter = 0
    for params, modelname, modeltype in modelset:
        if modeltype != 'nn':
            if modeltype !='logreg':
                if "cross" in modeltype:
                    if "full" in modelname.lower():
                            full_cross = True
                    model = Net_cross(parsed_shots, parsed_shift, n_qubits, num_layers, full_cross)
                else:
                    if '6' in modelname:
                        num_layers = 6
                    model = Net(parsed_shots, parsed_shift, n_qubits, num_layers, layerorder)
                model.eval()
                datacounter = 0
                for data in dataset:
                    if datacounter > maxdata:
                        break
                    losses = get_shift_loss(data, model, params, shift, modeltype)
                    baseloss = losses[0]
                    shiftlosses = losses[1:]
                    saliencylist = []
                    for item in shiftlosses:
                        saliency = (baseloss-item)/shift
                        # conversion from saliency of label to saliency of expectation value
                        input, label = data
                        label = remap(label.item())
                        if convert:
                            saliency = saliency*(-1/label)
                        saliencylist.append(saliency)
                    if signed:
                        df.loc[len(df)]=[datacounter,modelname]+saliencylist
                    else:
                        abslist = [abs(num) for num in saliencylist]
                        df.loc[len(df)]=[datacounter, modelname]+abslist
                    datacounter+=1
            else:
                df.loc[len(df)]=[0, 'Logistic Regression']+params.coef_.tolist()[0]
            modelcounter+=1

    df['datanum'] = df['datanum'].astype(int)

    return df

def nn_loss(X, y, datanum, model):
    pred = model.predict_proba(X[datanum:datanum+1])[0][1]
    return 1-(remap(y[datanum])*remap(pred))

def get_shift_loss_nn(X, y, datanum, model, shift):
    losses = []
    losses.append(nn_loss(X,y, datanum, model))
    datum = X[datanum]
    for i in range(len(datum)):
        Xcopy = X.copy()
        shifteddatum = datum.copy()
        shifteddatum[i] -= shift
        Xcopy[datanum] = shifteddatum
        shift_loss = nn_loss(Xcopy, y, datanum, model)
        losses.append(shift_loss)
    return losses

def get_saliency_nn(X, y, n_qubits, model, modelname, shift, maxdata,signed=False,convert=True):
    columns = []
    columns.append('datanum')
    columns.append("modelname")
    for n in range(n_qubits-1):
        columns.append("feature_"+str(n))
    df = pd.DataFrame(columns = columns)
    for datanum in range(len(X)):
        if datanum > maxdata:
            break
        losses = get_shift_loss_nn(X, y, datanum, model, shift)
        baseloss = losses[0]
        shiftlosses = losses[1:]
        saliencylist = []
        for item in shiftlosses:
            saliency = (baseloss-item)/shift
            # adding the conversion
            label = y[datanum]
            label = remap(label)
            if convert:
                saliency = saliency*(-1/label)
            saliencylist.append(saliency)
        paramsetcounter = 0
        if signed:
            df.loc[len(df)]=[paramsetcounter, str(modelname)]+saliencylist
        else:
            abslist = [abs(num) for num in saliencylist]
            df.loc[len(df)]=[paramsetcounter, str(modelname)]+abslist
    df['datanum'] = df['datanum'].astype(int)

    return df

def saliency_plot(saliency_df, title='Saliency comparison between models, averaged across data inputs', rescale=True):
    sliced = saliency_df[['modelname']+['feature_'+str(i) for i in range(len(saliency_df.columns)-2)]]
    avgd = sliced.groupby(['modelname']).mean()
    if rescale:
        avgd = avgd.div(avgd.abs().max(axis=1), axis=0)
    fig = avgd.plot.bar(title=title)
    fig.legend(bbox_to_anchor=(1.0, 1.0))



# %% logreg

def prep_plot_components_lr(X, y, X2, y2):
    ###the logistic regression###
    scores = []
    for i in range(10):
        clf = LogisticRegression(C=10**(1-i), penalty='l2', solver='liblinear')
        sc = cross_val_score(clf, X, y, cv=5)
        scores.append(sc.mean())

    scores_df = pd.DataFrame({'i': range(1, len(scores) + 1), 'score': scores})

    i_star = scores_df['score'].idxmax()

    clf_lr = LogisticRegression(C=10**(1-i_star), penalty='l2', solver='liblinear')


    clf_lr.fit(X, y)

    print(f'coefficients: {clf_lr.coef_}')
    print(f'intercept: {clf_lr.intercept_}')
    ### - - - - ###

    #getting fpr and tpr
    y_score = clf_lr.decision_function(X2)
    fpr, tpr, _ = roc_curve(y2, y_score)

    return (fpr, tpr, auc(fpr,tpr))

def split_labels(df):
    labelcol = len(df.columns)-1
    X = df.drop(str(labelcol), axis=1)
    y = df[str(labelcol)]
    return(X,y)

def get_logreg(X,y):
    ###the logistic regression###
    scores = []
    for i in range(10):
        clf = LogisticRegression(C=10**(1-i), penalty='l2', solver='liblinear')
        sc = cross_val_score(clf, X, y, cv=5)
        scores.append(sc.mean())

    scores_df = pd.DataFrame({'i': range(1, len(scores) + 1), 'score': scores})

    i_star = scores_df['score'].idxmax()

    clf_lr = LogisticRegression(C=10**(1-i_star), penalty='l2', solver='liblinear')


    clf_lr.fit(X, y)

    print(clf_lr.coef_)
    print(clf_lr.intercept_)
    ### - - - - ###
    return clf_lr

#%% param saliency defs

def qnn_params_shift_loss(data, model, params, shift, continuous=True):
    """returns a list of the unshifted loss followed the loss when each parameter is shifted independently (for qnn)"""
    input, label = data
    losses = []
    losses.append(model.eval_forward(params, continuous, data))
    for i in range(len(params)):
        shiftedparams = params.copy()
        shiftedparams[i] -= shift
        shift_loss = model.eval_forward(shiftedparams, continuous, data)
        losses.append(shift_loss)
    return losses

def get_param_saliency(dataset, n_qubits, modeltuple, shift, maxdata, parsed_shots, parsed_shift, num_layers, layerorder, full_cross, signed=False, convert=True):
    """
    Returns a dataframe which describes input saliencies across various (trained) QNN models. Arguments are as follows:
        'dataset': The data for which saliency is desired. Data is expected in the form (input, label), and dataset is expected to be a list of data tuples.
        'n_qubits': Number of qubits in the QNN, i.e. number of input values + 1
        'modelset': A list of final models for which the saliency is wanted. Each entry is a tuple with (model, modeltype, thetalist)
        'shift': Amount by which parameters should be shifted.
        'maxdata': Caps the number of datapoints for which the saliency is computed
    """

    params, modelname, modeltype = modeltuple
    if 'qnn' not in modeltype.lower(): #only meant to work with qnns
        pass
    else:
        if 'cross' in modeltype.lower():
            model = Net_cross(parsed_shots, parsed_shift, n_qubits, num_layers, full_cross)
            model.eval()
            all_sal = []
            datacounter = 0
            for data in dataset:
                if datacounter > maxdata:
                        break
                losses = qnn_params_shift_loss(data, model, params, shift, continuous=True)
                baseloss = losses[0]
                sal = []
                for item in losses[1:]:
                    saliency = (baseloss-item)/shift
                    input, label = data
                    label = remap(label.item())
                    if convert:
                        saliency = saliency*(-1/label)
                    if signed:
                        sal.append(saliency)
                    else:
                        sal.append(abs(saliency))
                all_sal.append(sal)
                datacounter+=1
            df = pd.DataFrame(all_sal)
            return df
        else:
            model = Net(parsed_shots, parsed_shift, n_qubits, num_layers, layerorder)
            model.eval()
            all_sal = []
            datacounter = 0
            for data in dataset:
                if datacounter > maxdata:
                        break
                losses = qnn_params_shift_loss(data, model, params, shift, continuous=True)
                baseloss = losses[0]
                sal = []
                for item in losses[1:]:
                    saliency = (baseloss-item)/shift
                    input, label = data
                    label = remap(label.item())
                    if convert:
                        saliency = saliency*(-1/label)
                    if signed:
                        sal.append(saliency)
                    else:
                        sal.append(abs(saliency))
                all_sal.append(sal)
                datacounter+=1
            df = pd.DataFrame(all_sal)
            return df

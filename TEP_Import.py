'''
File is used to import the TE dataset
'''
import numpy as np

FOLDER_PATH = "TE_Process/"
TRAINING_SETS = []
TEST_SETS = []

for i in range(22):
    TRAINING_SETS.append("d" + str(i).zfill(2) + ".dat")
    TEST_SETS.append("d" + str(i).zfill(2) + "_te.dat")

def importTrainingSets(sets_to_import = range(22)):
    '''
    Takes a sequence of integers from 0-21 and returns a zip
    of the TEP training dataset name and data
    '''
    if not hasattr(type(sets_to_import), '__iter__'):
        raise TypeError("Expected an iterable object")
    datasetnames = []
    datasets = []
    for val in sets_to_import:
        if not type(val) == int:
            raise TypeError("Expceted an integer value")
        if val > 21 or val < 0:
            raise ValueError("Expected an integer between 0 and 21")
        name = TRAINING_SETS[val]
        data = np.loadtxt(FOLDER_PATH+name)

        datasetnames.append(name)
        datasets.append(data)
    training_sets = zip(datasetnames,datasets)
    return(training_sets)

def importTestSets(sets_to_import = range(22)):
    '''
    Takes a sequence of integers from 0-21 and returns a zip
    of the TEP test dataset name and data
    '''
    if not hasattr(type(sets_to_import), '__iter__'):
        raise TypeError("Expected an iterable object")
    datasetnames = []
    datasets = []
    for val in sets_to_import:
        if not type(val) == int:
            raise TypeError("Expceted an integer value")
        if val > 21 or val < 0:
            raise ValueError("Expected an integer between 0 and 21")
        name = TEST_SETS[val]
        data = np.loadtxt(FOLDER_PATH+name)

        datasetnames.append(name)
        datasets.append(data)

    test_sets = zip(datasetnames,datasets)
    return(test_sets)
    

if __name__ == "__main__":
    print("This file cannot be run directly, import this module to obtain the",
          "datasets of the Tennessee Eastman process")

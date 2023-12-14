import os
import numpy as np
import shutil
import pandas as pd


def train_test_split(data_path = 'data', test_size = 0.15):
    print("########### Train Test Val Script started ###########")
    
    
    classes_dir = os.listdir(data_path)
    processed_dir = data_path

    for cls in classes_dir:
        # Creating partitions of the data after shuffeling
        print("$$$$$$$ Class Name " + cls + " $$$$$$$")
        src = processed_dir +"//" + cls  # Folder to copy images from

        allFileNames = os.listdir(src)
        np.random.shuffle(allFileNames)
        train_FileNames, test_FileNames = np.split(np.array(allFileNames), [int(len(allFileNames) * (1-test_size))])

        train_FileNames = [src + '//' + name for name in train_FileNames.tolist()]
        test_FileNames = [src + '//' + name for name in test_FileNames.tolist()]

        print('Total images: '+ str(len(allFileNames)))
        print('Training: '+ str(len(train_FileNames)))
        print('Testing: '+ str(len(test_FileNames)))

        # Creating Train / Val / Test folders (One time use)
        os.makedirs(data_path + '/train//' + cls, exist_ok=True)
        os.makedirs(data_path + '/test//' + cls, exist_ok=True)

        # Copy-pasting images
        for name in train_FileNames:
            shutil.copy(name, data_path + '/train//' + cls)

        for name in test_FileNames:
            shutil.copy(name, data_path + '/test//' + cls)

    print("########### Train Test Val Script Ended ###########")

if __name__=='__main__':
    train_test_split()

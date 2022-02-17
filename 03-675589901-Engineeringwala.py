# HW3
# Fatema Engineeringwala

import struct
from array import array
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

eta = 0.5
epsilon = 0.13
n = 60000

training_image_file='/Users/my/Documents/CS559 - Neural Networks/Code/HW3/train-images-idx3-ubyte'
training_label_file='/Users/my/Documents/CS559 - Neural Networks/Code/HW3/train-labels-idx1-ubyte'
testing_image_file='/Users/my/Documents/CS559 - Neural Networks/Code/HW3/t10k-images-idx3-ubyte'
testing_label_file='/Users/my/Documents/CS559 - Neural Networks/Code/HW3/t10k-labels-idx1-ubyte'


# Returns index of max val in 1x10 matrix
def label(val):
    for i in range(len(val)):
        if(val[i] == max(val)): return i

# Step Function
def u(val):
    v = np.copy(val)
    for i in range(len(val)):
        if(val[i] == max(val)): v[i] = 1
        else: v[i] = 0
    return v

# Returns 1x10 matrix for a label
def labelToMatrix(val):
    v = [0] * 10
    v[val] = 1
    return v

def main():
    # Reading the MNIST files
    with open(training_label_file, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = array("B", file.read())  

    with open(training_image_file, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        image_data = array("B", file.read())  

    with open(testing_label_file, 'rb') as file:
        magic, test_size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        test_labels = array("B", file.read())  

    with open(testing_image_file, 'rb') as file:
        magic, test_size, test_rows, test_cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        test_image_data = array("B", file.read())        
    
    # Converts images into 1x784 matrix
    images = []
    for i in range(size):
        images.append([0] * rows * cols)
    for i in range(size):
        img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
        images[i][:] = img
    test_images = []
    for i in range(size):
        test_images.append([0] * test_rows * test_cols)
    for i in range(size):
        img = np.array(test_image_data[i * test_rows * test_cols:(i + 1) * test_rows * test_cols])
        test_images[i][:] = img

    # size = size of training data
    weight_matrix = np.random.randn(10, 784)
    
    data = []
    omega = np.copy(weight_matrix)
    epoch = 0
    error = [0] * size
    # ETA
    while True:    
        for i in range(n):
            x = label(np.dot(weight_matrix, images[i]))
            if(x != labels[i]) :
                error[epoch] += 1
        # Use this line to print data dynamically after each epoch
        if(epoch==0):
            print("\nReport for Misclassification:")
            print('Epoch \t\t Misclassifications ')
        print(str(epoch)+"\t\t\t"+str(error[epoch]))
        data.append([epoch,error[epoch]])
        
        epoch += 1
        for i in range(n):
            var = (labelToMatrix(labels[i]) - u(np.dot(weight_matrix,images[i])))
            weight_matrix = weight_matrix + eta * np.dot(var.reshape(10, 1), np.transpose(images[i]).reshape(1, 784))
        if ((error[epoch-1]/n) <= epsilon): break
    # Prints the table on console
    # print("\nReport for misclassification=\n",tabulate(data,headers=['Epoch','Misclassification']))
    # To draw graph
    plt.title('Epoch vs misclassifications ')
    plt.xlabel('Epoch')
    plt.ylabel('Misclassifications')
    plt.plot([item[0] for item in data], [item[-1] for item in data])
    plt.show()
        
    # Calculate errors in test data
    errors = 0
    for i in range(test_size):
        x = label(np.dot(weight_matrix, test_images[i]))
        if(x != test_labels[i]) :
            errors+=1
        
    print("\nErrors in testing data (in "+str(test_size)+" samples) = "+str(errors))
    print("Percentage of Misclassification in testing data = "+str(errors*100/test_size))
    

if __name__=="__main__":main()
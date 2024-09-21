# Running Tests in Command Line

### Instruction Assumptions: 
 - Each set of instructions assume you have done the following:
    1. You have forked this repository and saved your local copy of it to a certain location on your computer.
    2. You are running the code through a shell program, or a terminal
    3. Your current working directory of your terminal session is "NN_Code_Implementations"
        - If it is not, you can change your current working directory by entering the following command:
            > cd "~/*path/to/local/copy*/NN_Code_Implementations"

## Commands to Navigate to Directories where Each Test File is Located 
- Directory where cross entropy cost function tests are located: "**NN_Code_Implementations**"
    > If you have followed the third aforementioned assumption, there is no command needed to change the current working directory to the aforementioned directory

- Directory where activation function tests are located: "**NN_Code_Implementations/activation_functions**"
    > cd activation_functions

- Directory where MLP tests are located: "**multilayer_perceptrons**"
    > "NN_Code_Implementations/

## How to Compile and Run Tests for Cross Entropy Cost Function

1. Type the following command to compile the associated testing file.
    > gcc cost_function_tests.c -o cost_function_tests

2. Type the following command to execute the tests for the implemented cost function.
    > ./cost_function_tests

2. To view the results of the test in the terminal, type the following command
    > less cost_function_tests.txt


## How to Run Tests for Activation Functions

1. Change your current working directory to the "activation_functions" directory
    > cd activation_functions

2. Type the following command to compile the associated testing file.
    > gcc function_tests.c -o function_tests

3. Type the following command to execute the tests for the implemented activation functions
    > ./function_tests

4. To view the results of the test in the terminal, type the following command
    > less function_tests.txt


## How to Run Tests for MLP 

1. Change your current working directory to the "multilayer_perceptron" directory
    > cd multilayer_perceptron

2. Type the following command to compile the associated testing file.
    > gcc mlp_tests.c -o mlp_tests

2. Type the following command to execute the tests for the implemented MLP
    > ./mlp_tests

3. To view the results of the test in the terminal, type the following command
    > less mlp_tests.txt


## How to Run Forward Pass Tests for All types of CNN Layers except the Fully-Connected Layer

1. Change your current working directory to the "cnn" directory
    > cd cnn

2. Type the following command to execute the tests for the implemented CNN layers
    > ./layer_tests

3. To view the results of the test in the terminal, type the following command
    > less layer_tests.txt


## How to Run Forward Pass Tests for Fully-Connected Layer

1. Change your current working directory to the "fully_connected" directory
    > cd fully_connected

2. Type the following command to execute the tests for the implemented fully-connected layer
    > ./full_conn_layer_tests

3. To view the results of the test in the terminal, type the following command
    > less full_conn_layer_tests.txt
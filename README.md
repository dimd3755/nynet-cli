# nynet-cli
![Static Badge](https://img.shields.io/badge/release-v0.8.1-yellow)
![Static Badge](https://img.shields.io/badge/license-Apache%202.0-yellow)
![Static Badge](https://img.shields.io/badge/build-passing-green)

This tool was created as an experiment project having to do with Artificial Intelligence, it is a lightweight CLI tool whose purpose is portability and data analysis

# Features
- **Lightweight & Dependency-Free** - Built entirely with C++ standard library (except CLI11 for command parsing and nlohmann/json for data serialization)
- **Portable CLI Interface** - Easy-to-use command-line tool with built-in help system
- **Flexible Network Architecture** - Support for multiple hidden layers with customizable sizes and activation functions
- **Three Activation Functions** - ReLU, Sigmoid, and Tanh
- **Complete Training Pipeline** - Create networks, train with data, and evaluate on test sets
- **JSONL Data Format** - Human-readable text format for training and test data
- **Logging Support** - Track training progress with loss metrics and model outputs
- **Binary State Persistence** - Save and load trained networks for reuse

# Credits
This tool was created using the listed dependencies:
* [nlohmann/json](https://github.com/nlohmann.json) for json format support
* [CLI11](https://github.com/cliutils/cli11) for CLI tool support

# Requirements
1. (*Optional*) [**Git**](https://git-scm.com/install/) for cloning the project.
2. [CMake](https://cmake.org/download/) for building the project.
3. C++ Compiler (eg. GNU, Clang, MSVC) for compiling the project into binary.

# Set up
First you need to clone the project with git, (or just download the [.zip](https://github.com/dimd3755/NeuralNetwork/archive/refs/heads/master.zip) file):

    git clone https://github.com/dimd3755/NeuralNetwork


Once you are inside the repository, use cmake to build the project:

    mkdir build

    cd build

    cmake ..

    cmake --build .

and you will find the executable inside build/ folder.

# Documentation

## Overview
`nynet-cli` is a command-line tool for creating, training, and testing neural networks. It provides three main subcommands:
- **`new`** - Creates and initializes a new neural network from a configuration file
- **`train`** - Trains an existing neural network using training data
- **`test`** - Evaluates a trained network on test data

## Getting Help
View help for the entire CLI or specific commands:
```bash
nynet --help              # View all available commands
nynet new --help          # View options for 'new' command
nynet train --help        # View options for 'train' command
nynet test --help         # View options for 'test' command
```

## Commands

### 1. `new` - Create a Neural Network
Creates a new neural network from a JSONL configuration file and saves its initial state.

**Syntax:**
```bash
nynet new -s <source> -d <destination>
```

**Options:**
- `-s, --src <path>` **(required)** - Path to the network configuration file (JSONL format)
- `-d, --dst <path>` **(required)** - Path where the network state will be saved (binary format)

**Configuration File Format (JSONL):**
```jsonl
{"inputs": 2, "hidden": [4, 4], "outputs": 1, "hidden_func": "relu", "outputs_func": "sigmoid", "lr": 0.01}
```

**Parameters:**
- `inputs` - Number of input neurons
- `hidden` - Array of hidden layer sizes (e.g., [4, 4] creates two hidden layers with 4 neurons each)
- `outputs` - Number of output neurons
- `hidden_func` - Activation function for hidden layers: `relu`, `sigmoid`, or `tanh`
- `outputs_func` - Activation function for output layer: `relu`, `sigmoid`, or `tanh`
- `lr` - Learning rate (typically 0.001 to 0.1)

**Example:**
```bash
nynet new -s network_config.jsonl -d network.bin
```

### 2. `train` - Train a Neural Network
Trains an existing neural network using training data and optionally saves logs.

**Syntax:**
```bash
nynet train -s <state> -f <file> [-e <epoch>] [-l <logs>]
```

**Options:**
- `-s, --state <path>` **(required)** - Path to the network state file (created with `new` command)
- `-f, --file <path>` **(required)** - Path to training data file (JSONL format)
- `-e, --epoch <number>` **(optional)** - Number of training epochs (default: 1)
- `-l, --logs <path>` **(optional)** - Path where training logs will be saved (JSONL format)

**Training Data Format (JSONL):**
Each line contains one training sample:
```jsonl
{"x": [0.0, 0.0], "y": [0.0]}
{"x": [0.0, 1.0], "y": [1.0]}
{"x": [1.0, 0.0], "y": [1.0]}
{"x": [1.0, 1.0], "y": [0.0]}
```

**Example:**
```bash
nynet train -s network.bin -f training_data.jsonl -e 1000 -l training_logs.jsonl
```

After training, the network state is automatically saved back to the state file.

### 3. `test` - Test a Neural Network
Evaluates a trained network on test data and generates predictions.

**Syntax:**
```bash
nynet test -s <state> -f <file> -o <output>
```

**Options:**
- `-s, --state <path>` **(required)** - Path to the trained network state file
- `-f, --file <path>` **(required)** - Path to test data file (JSONL format)
- `-o, --output <path>` **(required)** - Path where predictions will be saved (JSONL format)

**Test Data Format (JSONL):**
Each line contains one test sample (only input, no target):
```jsonl
{"x": [0.0, 0.0]}
{"x": [0.0, 1.0]}
{"x": [1.0, 0.0]}
{"x": [1.0, 1.0]}
```

**Output Format (JSONL):**
Predictions are saved with the same structure:
```jsonl
{"output": [0.05]}
{"output": [0.95]}
{"output": [0.92]}
{"output": [0.08]}
```

**Example:**
```bash
nynet test -s network.bin -f test_data.jsonl -o predictions.jsonl
```

## Example: XOR Problem

### Step 1: Create the network
Create `config.jsonl`:
```jsonl
{"inputs": 2, "hidden": [3, 3], "outputs": 1, "hidden_func": "relu", "outputs_func": "sigmoid", "lr": 0.001}
```

Create the network:
```bash
nynet new -s config.jsonl -d network.bin
```

### Step 2: Prepare training data
Create `training.jsonl`:
```jsonl
{"x": [0.0, 0.0], "y": [0.0]}
{"x": [0.0, 1.0], "y": [1.0]}
{"x": [1.0, 0.0], "y": [1.0]}
{"x": [1.0, 1.0], "y": [0.0]}
```

### Step 3: Train the network
```bash
nynet train -s network.bin -f training.jsonl -e 5000 -l training_logs.jsonl
```

### Step 4: Test the network
Create `test.jsonl`:
```jsonl
{"x": [0.0, 0.0]}
{"x": [0.0, 1.0]}
{"x": [1.0, 0.0]}
{"x": [1.0, 1.0]}
```

Run the test:
```bash
nynet test -s network.bin -f test.jsonl -o predictions.jsonl
```

View the predictions:
```bash
cat predictions.jsonl
```
(Should be close to their target values)

## Supported Activation Functions
- **ReLU** - Rectified Linear Unit: `relu`
- **Sigmoid** - Sigmoid: `sigmoid`
- **Tanh** - Hyperbolic Tangent: `tanh`

## Data Format Notes
- All numeric values use 64-bit floating point (double)
- JSONL format: JSON Lines (one JSON object per line)
- Although considered portable, has some limits in save/load functions (cpu architecture limit)
- Empty lines in JSONL files are automatically skipped

# Architecture

## Neural Network Structure
nynet-cli implements a standard feedforward artificial neural network with the following components:

### Neurons
The fundamental unit of the network:
- Each neuron has customizable weights and a bias term
- Weights are initialized using He initialization: $\text{weights} \sim \text{Uniform}(-\sqrt{\frac{2}{n}}, \sqrt{\frac{2}{n}})$ (where n is number of inputs)
- Supports linear combination followed by non-linear activation

### Layers
Organized collections of neurons:
- **Input Layer** - No activation, directly passes input values
- **Hidden Layers** - Multiple configurable layers with customizable activation functions
- **Output Layer** - Final layer with user-defined activation function

### Activation Functions
- **ReLU** - $f(x) = \max(0, x)$ - Used by default in hidden layers for non-linearity
- **Sigmoid** - $f(x) = \frac{1}{1 + e^{-x}}$ - Used by default in output layer for probability-like outputs
- **Tanh** - $f(x) = \tanh(x)$ - Alternative activation with range [-1, 1]

# License
This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) file for details.

### Quick Summary
- ✅ You can use, modify, and distribute this software
- ✅ Commercial use is permitted
- ✅ You must include a copy of the license
- ✅ You must state significant changes made to the code
- ❌ Liability and warranty are disclaimed

For the full legal text, see the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

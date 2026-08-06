#pragma once
#include "Neural.hpp"
#include <random>
#include <algorithm>

//'activate_type' enum:
enum class activate_type : int {
    RELU = 0,
    SIGMOID = 1,
};

//'Neuron' Struct:
Neuron::Neuron(const size_t& num_inputs) : weights(num_inputs) {
    std::random_device rd;
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = dist(rd);
    }
}

void Neuron::relu(const std::vector<double>& inputs) {
    double sum = 0.0;
    for (size_t i = 0; i < weights.size(); ++i) {
        sum += (weights[i] * inputs[i]);
    }
    sum += bias;

    output = std::max(0.0, sum);
}

void Neuron::sigmoid(const std::vector<double>& inputs) {
    double sum = 0.0;
    for (size_t i = 0; i < weights.size(); ++i) {
        sum += (weights[i] * inputs[i]);
    }
    sum += bias;

    output = 1.0 / (1 + std::exp(-sum));
}

void Neuron::update(const double& gradient, const double& learning_rate, const std::vector<double>& inputs) {
    bias -= gradient * learning_rate;
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] -= gradient * learning_rate * inputs[i];
    }
}

const double& Neuron::get_output() const {
    return output;
}

//'Layer' struct:
Layer::Layer(const size_t& num_neurons, const size_t& num_inputs, const activate_type& func) : function(func) {
    for (size_t i = 0; i < num_neurons; ++i) {
        neurons.push_back(Neuron(num_inputs));
    }

    outputs.resize(num_neurons, 0.0);
}

void Layer::forward(const std::vector<double>& inputs) {
    for (size_t i = 0; i < neurons.size(); ++i) {
        if (function == activate_type::RELU) neurons[i].relu(inputs);
        if (function == activate_type::SIGMOID) neurons[i].sigmoid(inputs);

        outputs[i] = neurons[i].get_output();
    }
}

const std::vector<double>& Layer::get_output() const {
    return outputs;
}

//'NeuralNetwork' class:
NeuralNetwork::NeuralNetwork(const size_t& num_inputs, const size_t& hidden_num_inputs, const size_t& num_outputs, const double& lr)
    : learning_rate(lr),
      hidden(hidden_num_inputs, num_inputs, activate_type::RELU),
      output(num_outputs, hidden_num_inputs, activate_type::SIGMOID) {};


void NeuralNetwork::forward(const std::vector<double>& inputs) {
    hidden.forward(inputs);
    output.forward(hidden.get_output());
}

void NeuralNetwork::train(const std::vector<double>& target, const std::vector<double>& inputs) {
    forward(inputs); // Give inputs
    std::vector<double> out_vec = output.get_output(); // Output of last layer
    std::vector<double> hid_vec = hidden.get_output(); // Output of hidden layer

    std::vector<double> out_grads(out_vec.size()); // Gradients of last layer
    for (size_t j = 0; j < out_grads.size(); ++j) {
        double error = out_vec[j] - target[j];
        out_grads[j] = error * out_vec[j] * (1.0 - out_vec[j]);
    }

    std::vector<double> hid_grads(hid_vec.size()); //Gradients of hidden layer
    for (size_t j = 0; j < hid_grads.size(); ++j) {
        if (hid_vec[j] <= 0.0) {
            hid_grads[j] = 0.0;
            continue;
        }

        double error = 0.0;
        for (size_t h = 0; h < out_grads.size(); ++h) {
            error += out_grads[h] * output.neurons[h].weights[j];
        }
        hid_grads[j] = error * 1.0;
    }

    //Update of last layer neurons
    for (size_t j = 0; j < out_grads.size(); ++j) {
        output.neurons[j].update(out_grads[j], learning_rate, hid_vec);
    }

    //Update of hidden layer neurons
    for (size_t j = 0; j < hid_grads.size(); ++j) {
        hidden.neurons[j].update(hid_grads[j], learning_rate, inputs);
    }
}

const std::vector<double>& NeuralNetwork::get_output() const {
    return output.get_output();
}
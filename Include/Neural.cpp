#include "Neural.hpp"
#include <random>
#include <algorithm>
#include <vector>

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

//Private functions:
double Neuron::linear(const std::vector<double>& inputs) {
    double sum = 0.0;
    for (size_t i = 0; i < weights.size(); ++i) {
        sum += (weights[i] * inputs[i]);
    }
    sum += bias;

    return sum;
}

//Public functions:
void Neuron::relu(const std::vector<double>& inputs) {
    output = std::max(0.0, linear(inputs));
}

void Neuron::sigmoid(const std::vector<double>& inputs) {
    output = 1.0 / (1 + std::exp(-linear(inputs)));
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
NeuralNetwork::NeuralNetwork(const size_t& input_num_neurons, const size_t& hidden_num_neurons, const size_t& output_num_neurons, const size_t& hidden_amount, const double& lr)
    : learning_rate(lr),
      hidden(),
      output(output_num_neurons, hidden_num_neurons, activate_type::SIGMOID) {
    
    hidden.reserve(hidden_amount);
    size_t current_num_neurons = input_num_neurons;
    for (size_t i = 0; i < hidden_amount; ++i)  {
        hidden.emplace_back(hidden_num_neurons, current_num_neurons, activate_type::RELU);
        current_num_neurons = hidden_num_neurons;
    }
}


void NeuralNetwork::forward(const std::vector<double>& inputs) {
    const std::vector<double>* current_inputs = &inputs;
    for (auto& l : hidden) {
        l.forward(*current_inputs);
        current_inputs = &l.get_output();
    }
    output.forward(*current_inputs);
}

void NeuralNetwork::train(const std::vector<double>& target, const std::vector<double>& inputs) {
    this->forward(inputs);

    std::vector<double> output_out = output.get_output();
    std::vector<double> output_out_grads(output_out.size());
    for (size_t i = 0; i < output_out_grads.size(); ++i) {
        double err = output_out[i] - target[i];
        output_out_grads[i] = err * output_out[i] * (1.0 - output_out[i]);
    }

    std::vector<std::vector<double>> hidden_outs(hidden.size());
    std::vector<std::vector<double>> hidden_outs_grads(hidden_outs.size());
    for (size_t i = 0; i < hidden_outs.size(); ++i) {
        hidden_outs[i] = hidden[i].get_output();
    }

    for (size_t i = 0; i < hidden_outs_grads.size(); ++i) {
        hidden_outs_grads[i].resize(hidden_outs[i].size(), 0.0);
    }

    for (long long i = static_cast<long long>(hidden_outs_grads.size()) - 1; i >= 0; --i) {
        for (size_t j = 0; j < hidden[i].neurons.size(); ++j) {
            double err_sum = 0.0;
            if (i == hidden_outs_grads.size() - 1) {
                for (size_t h = 0; h < output.neurons.size(); ++h) {
                    err_sum += output_out_grads[h] * output.neurons[h].weights[j];
                }
            } else {
                for (size_t h = 0; h < hidden[i+1].neurons.size(); ++h) {
                    err_sum += hidden_outs_grads[i+1][h] * hidden[i+1].neurons[h].weights[j];
                }
            }
            hidden_outs_grads[i][j] = err_sum;
        }
    }

    for (size_t i = 0; i < output.neurons.size(); ++i) {
        output.neurons[i].update(output_out_grads[i], learning_rate, hidden_outs.back());
    }

    for (size_t i = 0; i < hidden.size(); ++i) {
        for (size_t j = 0; j < hidden[i].neurons.size(); ++j) {
            if (i == 0) {
                hidden[i].neurons[j].update(hidden_outs_grads[i][j], learning_rate, inputs);
            } else {
                hidden[i].neurons[j].update(hidden_outs_grads[i][j], learning_rate, hidden_outs[i-1]);
            }
        }
    }
}

const std::vector<double>& NeuralNetwork::get_output() const {
    return output.get_output();
}
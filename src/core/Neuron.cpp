#include "Neural.hpp"
#include <random>

//Neuron Struct:
Neuron::Neuron(const size_t& num_inputs) : weights(num_inputs), bias(0.0) {
    if (num_inputs == 0) return;
    double limit = std::sqrt(2.0 / num_inputs);
    std::uniform_real_distribution<double> dist(-limit, limit);
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = dist(global_rng());
    }
}

double Neuron::linear(const std::vector<double>& inputs) {
    double sum = 0.0;
    if (inputs.size() != weights.size()) {
        std::cerr << "Neuron::linear: inputs.size() != weights.size()" << '\n';
        return 0.0;
    }
    for (size_t i = 0; i < weights.size(); ++i) {
        sum += (weights[i] * inputs[i]);
    }
    sum += bias;

    return sum;
}

void Neuron::relu(const std::vector<double>& inputs) {
    double z = linear(inputs);
    output = std::max(0.0, z);
}

void Neuron::sigmoid(const std::vector<double>& inputs) {
    double z = linear(inputs);
    output = 1.0 / (1 + std::exp(-z));
}

void Neuron::tanh(const std::vector<double>& inputs) {
    double z = linear(inputs);
    output = std::tanh(z);
}

void Neuron::update(const double& gradient, const double& learning_rate, const std::vector<double>& inputs) {
    if (inputs.size() != weights.size()) {
        std::cerr << "Neuron::update: inputs.size() != weights.size()" << '\n';
        return;
    }

    bias -= gradient * learning_rate;
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] -= gradient * learning_rate * inputs[i];
    }
}

bool Neuron::save(std::ostream& out) {
    out.write(reinterpret_cast<const char*>(&bias), sizeof(bias));

    size_t weights_size = weights.size();
    out.write(reinterpret_cast<const char*>(&weights_size), sizeof(weights_size));

    if (weights_size > 0) {
        out.write(reinterpret_cast<const char*>(weights.data()), weights_size * sizeof(weights[0]));
    }

    return out.good();
}

bool Neuron::load(std::istream& in) {
    const size_t MAX_WEIGHTS = 10000000;
    in.read(reinterpret_cast<char*>(&bias), sizeof(bias));

    size_t weights_size = 0;
    in.read(reinterpret_cast<char*>(&weights_size), sizeof(weights_size));

    if (weights_size > MAX_WEIGHTS) {
        std::cerr << "weights.size() exceeds the sane limit of 10.000.000," << '\n';
        return false;
    }

    weights.resize(weights_size);
    if (weights_size > 0) {
        in.read(reinterpret_cast<char*>(weights.data()), weights_size * sizeof(weights[0]));
    }

    return in.good();
}

const double& Neuron::get_output() const {
    return output;
}

void Neuron::set_output(double val) {
    output = val;
}
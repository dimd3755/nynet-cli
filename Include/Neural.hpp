#pragma once
#include <cmath>
#include <cstddef>
#include <vector>

enum class activate_type: int;

struct Neuron {
    private:
        double bias{};
        double output{};

        double linear(const std::vector<double>& inputs);

    public:
        std::vector<double> weights;

        Neuron(const size_t& num_inputs);

        void relu(const std::vector<double>& inputs);
        void sigmoid(const std::vector<double>& inputs);
        void update(const double& gradient, const double& learning_rate, const std::vector<double>& inputs);
        const double& get_output() const;
};

struct Layer {
    private:
        std::vector<double> outputs;
        activate_type function;

    public:
        std::vector<Neuron> neurons;

        Layer(const size_t& num_neurons, const size_t& num_inputs, const activate_type& func);

        void forward(const std::vector<double>& inputs);
        const std::vector<double>& get_output() const;
};

class NeuralNetwork {
    private:
        std::vector<Layer> hidden;
        Layer output;
        double learning_rate;

    public:
        NeuralNetwork(const size_t& input_num_neurons, const size_t& hidden_num_neuron, const size_t& output_num_neurons, const size_t& hidden_amount=1, const double& lr=0.01);

        void forward(const std::vector<double>& inputs);
        void train(const std::vector<double>& target, const std::vector<double>& inputs);
        const std::vector<double>& get_output() const;
};
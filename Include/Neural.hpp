#pragma once
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>
#include <fstream>
#include <random>

static_assert(sizeof(double) == 8, "Platform must use 64-bit doubles for binary storage");

static std::mt19937& global_rng() {
    thread_local static std::mt19937 gen;
    thread_local static bool initialized = false;
    if (!initialized) {
        gen.seed(std::random_device{}());
        initialized = true;
    }
    return gen;
}

enum class activate_type : int {
    RELU = 0,
    SIGMOID = 1,
    TANH = 2,
};

struct Logln {
    std::vector<double> target;
    std::vector<double> output;
    double loss;
    size_t epoch;

    bool finished_successfully;
};

struct Neuron {
    private:
        double bias{};
        double output{};

    public:
        std::vector<double> weights;

        Neuron(const size_t& num_inputs);

        double linear(const std::vector<double>& inputs);
        void relu(const std::vector<double>& inputs);
        void sigmoid(const std::vector<double>& inputs);
        void tanh(const std::vector<double>& inputs);

        void update(const double& gradient, const double& learning_rate, const std::vector<double>& inputs);

        bool save(std::ostream& out);
        bool load(std::istream& in);

        const double& get_output() const;
        void set_output(double val);
};

struct Layer {
    private:
        std::vector<double> outputs;
        activate_type function;

    public:
        std::vector<Neuron> neurons;

        Layer(const size_t& num_neurons, const size_t& num_inputs, const activate_type& func);

        void forward(const std::vector<double>& inputs);

        bool save(std::ostream& out);
        bool load(std::istream& in);

        const std::vector<double>& get_output() const;
};

class NeuralNetwork {
    private:
        std::vector<Layer> hidden;
        Layer output;
        double learning_rate;
        activate_type hfunc;
        activate_type ofunc;

        double derive(const activate_type& func, const double& output);
        double gradient(const activate_type& func, const double& output, const double& target);

    public:
        NeuralNetwork() : output(0, 0, activate_type::SIGMOID), learning_rate(0.001), hfunc(activate_type::RELU), ofunc(activate_type::SIGMOID) {}

        NeuralNetwork(const size_t& input_num_neurons, const std::vector<size_t>& hidden_neuron_sizes, const size_t& output_num_neurons,
             const activate_type& hidden_func=activate_type::RELU, const activate_type& output_func=activate_type::SIGMOID, const double& lr=0.001);

        void forward(const std::vector<double>& inputs) ;
        Logln train(const std::vector<double>& inputs, const std::vector<double>& target);
        bool save(std::ofstream& file_out);
        bool load(std::ifstream& file_in);
        const std::vector<double>& get_output() const;
};
#include "Neural.hpp"

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
        if (function == activate_type::TANH) neurons[i].tanh(inputs);
        outputs[i] = neurons[i].get_output();
    }
}

bool Layer::save(std::ostream& out) {
    int casted_function = static_cast<int>(function);
    out.write(reinterpret_cast<const char*>(&casted_function), sizeof(casted_function));

    size_t neurons_size = neurons.size();
    out.write(reinterpret_cast<const char*>(&neurons_size), sizeof(neurons_size));

    for (auto& n: neurons) {
        if (!n.save(out)) return false;
    } 
    return out.good();
}

bool Layer::load(std::istream& in) {
    const size_t MAX_NEURONS = 1000000;
    
    int casted_function;
    in.read(reinterpret_cast<char*>(&casted_function), sizeof(casted_function));

    function = static_cast<activate_type>(casted_function);

    size_t neurons_size = 0;
    in.read(reinterpret_cast<char*>(&neurons_size), sizeof(neurons_size));
    
    if (neurons_size > MAX_NEURONS) {
        std::cerr << "neurons.size() exceeds sane limit of 1.000.000." << '\n';
        return false;
    }

    neurons.clear();
    neurons.reserve(neurons_size);
    for (size_t i = 0; i < neurons_size; ++i) {
        Neuron n(0);
        if (!n.load(in)) return false;
        neurons.push_back(n);
    } 

    outputs.resize(neurons_size, 0.0);
    return in.good();
}

const std::vector<double>& Layer::get_output() const {
    return outputs;
}
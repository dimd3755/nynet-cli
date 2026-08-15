#include "Neural.hpp"

//'NeuralNetwork' class:
NeuralNetwork::NeuralNetwork(const size_t& input_num_neurons, const std::vector<size_t>& hidden_neuron_sizes, const size_t& output_num_neurons,
     const activate_type& hidden_func, const activate_type& output_func, const double& lr)
    : learning_rate(lr),
      hfunc(hidden_func),
      ofunc(output_func),
      hidden(),
      output(output_num_neurons, hidden_neuron_sizes.size() > 0 ? hidden_neuron_sizes.back() : input_num_neurons, output_func) {
    
    hidden.reserve(hidden_neuron_sizes.size());
    size_t current_num_neurons = input_num_neurons;
    for (size_t i = 0; i < hidden_neuron_sizes.size(); ++i)  {
        hidden.emplace_back(hidden_neuron_sizes[i], current_num_neurons, hidden_func);
        current_num_neurons = hidden_neuron_sizes[i];
    }
}

//Private functions:
double NeuralNetwork::derive(const activate_type& func, const double& output) {
    double derivative = 0.0;
    if (func == activate_type::RELU) {
        derivative = (output > 0.0) ? 1.0 : 0.0;
    } else if (func == activate_type::SIGMOID) {
        derivative = output * (1.0 - output);
    } else if (func == activate_type::TANH) {
        derivative = 1.0 - (output * output);
    }

    return derivative;
}

double NeuralNetwork::gradient(const activate_type& func, const double& output, const double& target) {
    double err = output - target;
    double derivative = derive(func, output);

    return err * derivative;
}

//Public functions:
void NeuralNetwork::forward(const std::vector<double>& inputs) {
    const std::vector<double>* current_inputs = &inputs;
    for (auto& l : hidden) {
        l.forward(*current_inputs);
        current_inputs = &l.get_output();
    }
    output.forward(*current_inputs);
}

Logln NeuralNetwork::train(const std::vector<double>& inputs, const std::vector<double>& target) {
    Logln logln;
    logln.finished_successfully = true; // setting 'finished_successfully' bool to true by default
    logln.target = target; // (1/4) note target inside logs (finished_successfully doesnt count)
    this->forward(inputs);

    if (target.size() != output.get_output().size()) {
        std::cerr << "NeuralNetwork::train: target.size() != output.get_output().size()" << '\n';
        logln.finished_successfully = false;
        return logln;
    }

    // Outputs and output gradients
    std::vector<double> output_out = output.get_output(); // Fills output_out with the outputs of the output layer
    std::vector<double> output_out_grads(output_out.size());

    logln.output = output_out; // (2/4) note output inside logs

    // Filling output gradients
    std::vector<double> errors; // capturing output errors (for logln notes)
    errors.reserve(output_out.size());
    for (size_t i = 0; i < output_out_grads.size(); ++i) {
        output_out_grads[i] = gradient(ofunc, output_out[i], target[i]);
        errors.push_back(output_out[i] - target[i]);
    }

    // Getting average loss for logln.loss
    double err_sum = 0.0;
    for (auto& err: errors) {
        err_sum += err;
    }

    logln.loss = err_sum / errors.size(); //(3/4) noted average loss

    // Hidden and hidden gradients
    std::vector<std::vector<double>> hidden_outs(hidden.size());
    std::vector<std::vector<double>> hidden_outs_grads(hidden.size());

    // Get outputs of hidden layer
    for (size_t i = 0; i < hidden_outs.size(); ++i) {
        hidden_outs[i] = hidden[i].get_output();
    }

    // Resize gradient vectors to match hidden layer output's size and fill them with 0.0
    for (size_t i = 0; i < hidden_outs_grads.size(); ++i) {
        hidden_outs_grads[i].resize(hidden_outs[i].size(), 0.0);
    }

    // Filling the hidden gradients
    for (long long i = static_cast<long long>(hidden_outs_grads.size()) - 1; i >= 0; --i) { // For ever layer (backwards)
        for (size_t j = 0; j < hidden[i].neurons.size(); ++j) { // For every neuron in layer
            double err_sum = 0.0;
            if (i == hidden_outs_grads.size() - 1) { // If this hidden layer has output layer infront of it
                for (size_t h = 0; h < output.neurons.size(); ++h) {
                    err_sum += output_out_grads[h] * output.neurons[h].weights[j];
                }
            } else { // If this hidden layer has a hidden layer infront of it
                for (size_t h = 0; h < hidden[i+1].neurons.size(); ++h) {
                    err_sum += hidden_outs_grads[i+1][h] * hidden[i+1].neurons[h].weights[j];
                }
            }
            double derivative = derive(hfunc, hidden_outs[i][j]);
            hidden_outs_grads[i][j] = err_sum * derivative;
        }
    }

    // Updates
    // Updating output layer
    if (hidden.empty()) { // Case where the next layer after the current hidden is the output layer
        for (size_t i = 0; i < output.neurons.size(); ++i) {
            output.neurons[i].update(output_out_grads[i], learning_rate, inputs);
        }   
        return logln; // Because it has no hidden layers to update after that
    } else { // Case where the next layer after the current hidden is another hidden layer
        for (size_t i = 0; i < output.neurons.size(); ++i) {
            output.neurons[i].update(output_out_grads[i], learning_rate, hidden_outs.back());
        }  
    }

    // Updating hidden layers
    for (size_t i = 0; i < hidden.size(); ++i) {
        for (size_t j = 0; j < hidden[i].neurons.size(); ++j) {
            if (i == 0) { // Changes with inputs being the input layer
                hidden[i].neurons[j].update(hidden_outs_grads[i][j], learning_rate, inputs);
            } else { // Changes with inputs being the output of the previous hidden layer
                hidden[i].neurons[j].update(hidden_outs_grads[i][j], learning_rate, hidden_outs[i-1]);
            }
        }
    }

    return logln; // Epoch index variable will be handled by the main.cpp implementation
}

bool NeuralNetwork::save(std::ofstream& file_out) {
    file_out.write(reinterpret_cast<const char*>(&learning_rate), sizeof(learning_rate));

    int casted_hfunc = static_cast<int>(hfunc);
    int casted_ofunc = static_cast<int>(ofunc);

    file_out.write(reinterpret_cast<const char*>(&casted_hfunc), sizeof(casted_hfunc));
    file_out.write(reinterpret_cast<const char*>(&casted_ofunc), sizeof(casted_ofunc));
    
    size_t hidden_size = hidden.size();
    file_out.write(reinterpret_cast<const char*>(&hidden_size), sizeof(hidden_size));

    for (auto& layer: hidden) {
        if (!layer.save(file_out)) return false;
    }

    if (!output.save(file_out)) return false;
    return file_out.good();
}

bool NeuralNetwork::load(std::ifstream& file_in) {
    const size_t MAX_HIDDENS = 1000;
    file_in.read(reinterpret_cast<char*>(&learning_rate), sizeof(learning_rate));

    int casted_hfunc;
    int casted_ofunc;

    file_in.read(reinterpret_cast<char*>(&casted_hfunc), sizeof(casted_hfunc));
    file_in.read(reinterpret_cast<char*>(&casted_ofunc), sizeof(casted_ofunc));

    hfunc = static_cast<activate_type>(casted_hfunc);
    ofunc = static_cast<activate_type>(casted_ofunc);

    size_t hidden_size = 0;
    file_in.read(reinterpret_cast<char*>(&hidden_size), sizeof(hidden_size));

    if (hidden_size > MAX_HIDDENS) {
        std::cerr << "hidden_size exceeds the sane limit of 1000." << '\n';
        return false;
    }

    hidden.clear();
    hidden.reserve(hidden_size);
    for (size_t i = 0; i < hidden_size; ++i) {
        Layer layer(0, 0, hfunc);
        if (!layer.load(file_in)) return false;
        hidden.push_back(layer);
    }

    if (!output.load(file_in)) return false;
    return file_in.good();
}

const std::vector<double>& NeuralNetwork::get_output() const {
    return output.get_output();
}
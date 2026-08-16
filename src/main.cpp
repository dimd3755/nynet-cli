#include "nlohmann/json.hpp"
#include "CLI/CLI11.hpp"
#include "Neural.hpp"
#include <cctype>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::json;

bool stream_file(const std::string& src, std::vector<std::string>& dst);
activate_type string_to_activate_type(const std::string& src);
void logln_to_json(json& j, const Logln& l);

int main(int argc, char** argv) {
    CLI::App app{};

    app.set_help_all_flag();

    auto subc_new = app.add_subcommand("new", "Creates and saves state for the neural network");
    std::string src_file;
    std::string dst_file;

    subc_new->add_option("-s,--src", src_file, "the source instructions of the network (eg. input & output size, functions, lr etc.)")->required();
    subc_new->add_option("-d,--dst", dst_file, "the network's state in binary format which can be read by the network")->required();

    auto subc_train = app.add_subcommand("train", "The training function of the network");
    std::string state_file;
    std::string data_file;
    size_t epoch = 1;
    std::string logs_file;

    subc_train->add_option("-s,--state", state_file, "the network's state in binary format which contains the neurons settings")->required();
    subc_train->add_option("-f,--file", data_file, "the training data file in jsonl format which contains x and y tensors in same line")->required();
    subc_train->add_option("-e,--epoch", epoch, "the training epoch amount")->required(false);
    subc_train->add_option("-l,--logs", logs_file, "the training logs eg (loss overtime, outputs, etc.)")->required(false);

    auto subc_test = app.add_subcommand("test", "The main testing function of the network");
    std::string state_file2;
    std::string data_file2;
    std::string output_file;

    subc_test->add_option("-s,--state", state_file2, "the network's state in binary format which contains the neurons settings")->required();
    subc_test->add_option("-f,--file", data_file2, "the test data in jsonl format which contains x tensor")->required();
    subc_test->add_option("-o,--output", output_file, "the output which contains the result of the given data")->required();

    CLI11_PARSE(app, argc, argv);

    if (subc_new->parsed()) {
        std::vector<std::string> dst;
        if (!stream_file(src_file, dst)) {
            std::cerr << "Failed to stream src file." << '\n';
            return 1;
        }

        if (dst.size() != 1) {
            std::cerr << "There should be only 1 state of network in source file." << '\n';
            return 1;
        }

        try {
            std::ofstream dst_file_stream(dst_file);              
            if (!dst_file_stream.is_open()) {
                std::cerr << "Failed to save state." << '\n';
                return 1;
            }

            for (auto& ln: dst) {
                json j = json::parse(ln);

                auto inputs = j["inputs"].get<size_t>();
                auto hidden = j["hidden"].get<std::vector<size_t>>();
                auto outputs = j["outputs"].get<size_t>();
                auto hidden_func = j["hidden_func"].get<std::string>();
                auto outputs_func = j["outputs_func"].get<std::string>();
                auto lr = j["lr"].get<double>();
                
                NeuralNetwork nn(inputs, hidden, outputs, string_to_activate_type(hidden_func), string_to_activate_type(outputs_func), lr);
                nn.save(dst_file_stream);
            }
        } catch (std::exception& e) {
            std::cerr << e.what() << '\n';
            return 1; 
        }

    } else if (subc_train->parsed()) {
        std::ifstream state_file_istream(state_file);
        if (!state_file_istream.is_open()) {
            std::cerr << "Failed to open state file." << '\n';
            return 1;
        }

        NeuralNetwork nn;
        nn.load(state_file_istream);

        std::vector<std::string> dst;
        if (!stream_file(data_file, dst)) {
            std::cerr << "Failed to stream data file" << '\n';
            return 1;
        }

        try {
            std::ofstream logs_file_stream;
            if (subc_train->count("--logs") > 0) {
                logs_file_stream.open(logs_file);
                if (!logs_file_stream.is_open()) {
                    std::cerr << "Failed to open logs file." << '\n';
                    return 1;
                }
            }

            for (auto& ln: dst) {
                    json j = json::parse(ln);
                    auto x = j["x"].get<std::vector<double>>();
                    auto y = j["y"].get<std::vector<double>>();
                for (size_t i = 0; i < epoch; ++i) {
                    Logln log_res = nn.train(x, y);
                    
                    if (!log_res.finished_successfully) {
                        std::cerr << "Failed to train set " << i << ", quitting." << '\n';
                        return 1;
                    }

                    log_res.epoch = i;

                    if (logs_file_stream.is_open()) {
                        json json_log_res;
                        logln_to_json(json_log_res, log_res);

                        logs_file_stream << json_log_res.dump() << '\n';
                    }               
                }
            }
        } catch (std::exception& e) {
            std::cerr << e.what() << '\n';
            return 1;
        }
        std::ofstream state_file_ostream(state_file);      
        if (!state_file_ostream.is_open()) {
            std::cerr << "Failed to open state file for saving." << '\n';
            return 1;
        }
        nn.save(state_file_ostream);

    } else if (subc_test->parsed()) {
        std::ifstream state_file2_stream(state_file2);
        if (!state_file2_stream.is_open()) {
            std::cerr << "Failed to open state file." << '\n';
            return 1;
        }

        NeuralNetwork nn;
        nn.load(state_file2_stream);

        std::vector<std::string> dst;
        if (!stream_file(data_file2, dst)) {
            std::cerr << "Failed to stream." << '\n';
            return 1;
        }

        std::ofstream output_file_stream(output_file);

        if (!output_file_stream.is_open()) {
            std::cerr << "Failed to open output file." << '\n';
            return 1;
        }

        for (auto& ln: dst) {
            json j = json::parse(ln);

            auto x = j["x"].get<std::vector<double>>();

            nn.forward(x);

            json j2 = json{{"output", nn.get_output()}};

            for (auto& c: nn.get_output()) { //DEBUG
                std::cout << c << '\n';
            } // END OF DEBUG

            output_file_stream << j2.dump() << '\n';
        }
    }

    return 0;
}

bool stream_file(const std::string& src_path, std::vector<std::string>& dst) {
    std::ifstream src(src_path);

    if (!src.is_open()) {
        return false;
    }

    std::string line;
    while (std::getline(src, line)) {
        if (line.empty()) continue;
        dst.push_back(std::move(line));
    }

    return true;
}

activate_type string_to_activate_type(const std::string& src) {
    std::string lower = "";
    for (auto& c: src) {
        char lc = tolower(c);
        lower += lc;
    }

    if (lower == "relu") {
        return activate_type::RELU;
    } else if (lower == "sigmoid") {
        return activate_type::SIGMOID;
    } else if (lower == "tanh") {
        return activate_type::TANH;
    } else {
        return activate_type::RELU;
    }
}

void logln_to_json(json& j, const Logln& l) {
    j = json{{"target", l.target}, {"output", l.output}, {"loss", l.loss}, {"epoch_index", l.epoch}};
}

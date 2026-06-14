#include "tublex_runtime.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " <tublex_window_features_csv> [suspicion_threshold] [monitoring_threshold]\n\n"
                  << "Example:\n"
                  << "  " << argv[0] << " ../examples/sample_mixed_windows.csv\n"
                  << "  " << argv[0] << " ../examples/sample_mixed_windows.csv 0.60 0.35\n";

        return EXIT_FAILURE;
    }

    const std::string csv_path = argv[1];

    tublex::RuntimeConfig config;

    if (argc >= 3)
    {
        config.suspicion_threshold = std::stod(argv[2]);
    }

    if (argc >= 4)
    {
        config.monitoring_threshold = std::stod(argv[3]);
    }

    try
    {
        const auto windows = tublex::load_tublex_windows_csv(csv_path);
        const tublex::TublexRuntime runtime(config);

        std::cout << "Input file: " << csv_path << "\n";
        std::cout << "Suspicion threshold: " << config.suspicion_threshold << "\n";
        std::cout << "Monitoring threshold: " << config.monitoring_threshold << "\n\n";

        std::cout << "TUBLEX Embedded Runtime Skeleton\n";
        std::cout << "--------------------------------\n";

        std::vector<tublex::RuntimeResult> results;
        results.reserve(windows.size());

        for (const auto& window : windows)
        {
            const auto result = runtime.evaluate(window);
            results.push_back(result);
            tublex::print_runtime_result(result, window);
        }

        tublex::print_runtime_summary(results);
    }
    catch (const std::exception& error)
    {
        std::cerr << "Runtime error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

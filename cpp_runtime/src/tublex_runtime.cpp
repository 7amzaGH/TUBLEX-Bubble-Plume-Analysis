#include "tublex_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace tublex
{

namespace
{

std::string trim(const std::string& value)
{
    const std::string whitespace = " \t\n\r\f\v";
    const std::size_t start = value.find_first_not_of(whitespace);

    if (start == std::string::npos)
    {
        return "";
    }

    const std::size_t end = value.find_last_not_of(whitespace);
    return value.substr(start, end - start + 1);
}

std::vector<std::string> split_csv_line(const std::string& line)
{
    std::vector<std::string> tokens;
    std::stringstream stream(line);
    std::string token;

    while (std::getline(stream, token, ','))
    {
        tokens.push_back(trim(token));
    }

    return tokens;
}

double clamp01(double value)
{
    if (value < 0.0)
    {
        return 0.0;
    }

    if (value > 1.0)
    {
        return 1.0;
    }

    return value;
}

double normalized(double value, double scale)
{
    if (scale <= 0.0)
    {
        return 0.0;
    }

    return clamp01(value / scale);
}

bool has_column(const std::unordered_map<std::string, std::size_t>& columns, const std::string& name)
{
    return columns.find(name) != columns.end();
}

double get_double(
    const std::vector<std::string>& tokens,
    const std::unordered_map<std::string, std::size_t>& columns,
    const std::string& name
)
{
    const auto it = columns.find(name);

    if (it == columns.end())
    {
        throw std::runtime_error("Missing required column: " + name);
    }

    if (it->second >= tokens.size())
    {
        throw std::runtime_error("Invalid row: missing value for column " + name);
    }

    const std::string value = trim(tokens[it->second]);

    if (value.empty())
    {
        return 0.0;
    }

    return std::stod(value);
}

int get_int_optional(
    const std::vector<std::string>& tokens,
    const std::unordered_map<std::string, std::size_t>& columns,
    const std::string& name,
    int default_value
)
{
    const auto it = columns.find(name);

    if (it == columns.end() || it->second >= tokens.size())
    {
        return default_value;
    }

    const std::string value = trim(tokens[it->second]);

    if (value.empty())
    {
        return default_value;
    }

    return std::stoi(value);
}

void check_required_columns(const std::unordered_map<std::string, std::size_t>& columns)
{
    const std::vector<std::string> required_columns = {
        "mean_bubble_count",
        "max_bubble_count",
        "std_bubble_count",
        "continuity_ratio",
        "mean_vertical_chain",
        "temporal_variance",
        "prev3_mean_bubble_count",
        "prev3_std_bubble_count",
        "prev3_continuity_ratio",
        "prev3_mean_vertical_chain",
        "leak_evolution"
    };

    for (const auto& name : required_columns)
    {
        if (!has_column(columns, name))
        {
            throw std::runtime_error("Missing required column: " + name);
        }
    }
}

}  // namespace

TublexRuntime::TublexRuntime(const RuntimeConfig& config)
    : config_(config)
{
    if (config_.suspicion_threshold < 0.0 || config_.suspicion_threshold > 1.0)
    {
        throw std::invalid_argument("Suspicion threshold must be in the range [0, 1].");
    }

    if (config_.monitoring_threshold < 0.0 || config_.monitoring_threshold > 1.0)
    {
        throw std::invalid_argument("Monitoring threshold must be in the range [0, 1].");
    }
}

RuntimeResult TublexRuntime::evaluate(const TublexWindow& window) const
{
    RuntimeResult result;
    result.window_id = window.window_id;
    result.suspicion_score = compute_suspicion_score(window);

    if (result.suspicion_score >= config_.suspicion_threshold)
    {
        result.state = SuspicionState::SUSPICIOUS;
    }
    else if (result.suspicion_score >= config_.monitoring_threshold)
    {
        result.state = SuspicionState::MONITORING;
    }
    else
    {
        result.state = SuspicionState::NORMAL;
    }

    return result;
}

RuntimeConfig TublexRuntime::config() const
{
    return config_;
}

double TublexRuntime::compute_suspicion_score(const TublexWindow& window) const
{
    // Lightweight embedded-style scoring interface.
    //
    // This is intentionally a simple deterministic runtime skeleton.
    // It is not the trained Python Random Forest or XGBoost model.
    // The weights emphasize the same descriptor families used by TUBLEX:
    // density, temporal variability, continuity, vertical structure, and short-term evolution.
    const double density_score =
        0.15 * normalized(window.mean_bubble_count, 15.0) +
        0.10 * normalized(window.max_bubble_count, 25.0);

    const double variability_score =
        0.25 * normalized(window.temporal_variance, 35.0) +
        0.15 * normalized(window.std_bubble_count, 6.0);

    const double continuity_score =
        0.20 * clamp01(window.continuity_ratio);

    const double chain_score =
        0.10 * normalized(window.mean_vertical_chain, 5.0);

    const double evolution_score =
        0.05 * normalized(window.leak_evolution, 10.0);

    return clamp01(
        density_score +
        variability_score +
        continuity_score +
        chain_score +
        evolution_score
    );
}

std::vector<TublexWindow> load_tublex_windows_csv(const std::string& csv_path)
{
    std::ifstream file(csv_path);

    if (!file.is_open())
    {
        throw std::runtime_error("Could not open CSV file: " + csv_path);
    }

    std::string header_line;

    if (!std::getline(file, header_line))
    {
        throw std::runtime_error("CSV file is empty: " + csv_path);
    }

    const auto header_tokens = split_csv_line(header_line);

    std::unordered_map<std::string, std::size_t> columns;

    for (std::size_t i = 0; i < header_tokens.size(); ++i)
    {
        columns[header_tokens[i]] = i;
    }

    check_required_columns(columns);

    std::vector<TublexWindow> windows;
    std::string line;
    int row_index = 0;

    while (std::getline(file, line))
    {
        if (trim(line).empty())
        {
            continue;
        }

        const auto tokens = split_csv_line(line);

        TublexWindow window;
        window.window_id = get_int_optional(tokens, columns, "window_id", row_index);

        window.mean_bubble_count = get_double(tokens, columns, "mean_bubble_count");
        window.max_bubble_count = get_double(tokens, columns, "max_bubble_count");
        window.std_bubble_count = get_double(tokens, columns, "std_bubble_count");
        window.continuity_ratio = get_double(tokens, columns, "continuity_ratio");
        window.mean_vertical_chain = get_double(tokens, columns, "mean_vertical_chain");
        window.temporal_variance = get_double(tokens, columns, "temporal_variance");

        window.prev3_mean_bubble_count = get_double(tokens, columns, "prev3_mean_bubble_count");
        window.prev3_std_bubble_count = get_double(tokens, columns, "prev3_std_bubble_count");
        window.prev3_continuity_ratio = get_double(tokens, columns, "prev3_continuity_ratio");
        window.prev3_mean_vertical_chain = get_double(tokens, columns, "prev3_mean_vertical_chain");
        window.leak_evolution = get_double(tokens, columns, "leak_evolution");

        windows.push_back(window);
        ++row_index;
    }

    return windows;
}

std::string to_string(SuspicionState state)
{
    switch (state)
    {
        case SuspicionState::NORMAL:
            return "NORMAL";
        case SuspicionState::MONITORING:
            return "MONITORING";
        case SuspicionState::SUSPICIOUS:
            return "SUSPICIOUS";
        default:
            return "NORMAL";
    }
}

void print_runtime_result(const RuntimeResult& result, const TublexWindow& window)
{
    std::cout << std::fixed << std::setprecision(3);

    std::cout << "Window " << result.window_id
              << " | score: " << result.suspicion_score
              << " | state: " << to_string(result.state)
              << " | mean_count: " << window.mean_bubble_count
              << " | std_count: " << window.std_bubble_count
              << " | continuity: " << window.continuity_ratio
              << " | vertical_chain: " << window.mean_vertical_chain
              << "\n";
}

void print_runtime_summary(const std::vector<RuntimeResult>& results)
{
    std::size_t normal_count = 0;
    std::size_t monitoring_count = 0;
    std::size_t suspicious_count = 0;

    for (const auto& result : results)
    {
        if (result.state == SuspicionState::NORMAL)
        {
            ++normal_count;
        }
        else if (result.state == SuspicionState::MONITORING)
        {
            ++monitoring_count;
        }
        else if (result.state == SuspicionState::SUSPICIOUS)
        {
            ++suspicious_count;
        }
    }

    std::cout << "\nRuntime Summary\n";
    std::cout << "---------------\n";
    std::cout << "Total windows: " << results.size() << "\n";
    std::cout << "NORMAL: " << normal_count << "\n";
    std::cout << "MONITORING: " << monitoring_count << "\n";
    std::cout << "SUSPICIOUS: " << suspicious_count << "\n";
}

}  // namespace tublex

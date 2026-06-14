#ifndef TUBLEX_RUNTIME_HPP
#define TUBLEX_RUNTIME_HPP

#include <cstddef>
#include <string>
#include <vector>

namespace tublex
{

struct TublexWindow
{
    int window_id = 0;

    double mean_bubble_count = 0.0;
    double max_bubble_count = 0.0;
    double std_bubble_count = 0.0;
    double continuity_ratio = 0.0;
    double mean_vertical_chain = 0.0;
    double temporal_variance = 0.0;

    double prev3_mean_bubble_count = 0.0;
    double prev3_std_bubble_count = 0.0;
    double prev3_continuity_ratio = 0.0;
    double prev3_mean_vertical_chain = 0.0;
    double leak_evolution = 0.0;
};

enum class SuspicionState
{
    NORMAL,
    MONITORING,
    SUSPICIOUS
};

struct RuntimeConfig
{
    double suspicion_threshold = 0.60;
    double monitoring_threshold = 0.35;
};

struct RuntimeResult
{
    int window_id = 0;
    double suspicion_score = 0.0;
    SuspicionState state = SuspicionState::NORMAL;
};

class TublexRuntime
{
public:
    explicit TublexRuntime(const RuntimeConfig& config = RuntimeConfig());

    RuntimeResult evaluate(const TublexWindow& window) const;

    RuntimeConfig config() const;

private:
    RuntimeConfig config_;

    double compute_suspicion_score(const TublexWindow& window) const;
};

std::vector<TublexWindow> load_tublex_windows_csv(const std::string& csv_path);

std::string to_string(SuspicionState state);

void print_runtime_result(const RuntimeResult& result, const TublexWindow& window);

void print_runtime_summary(const std::vector<RuntimeResult>& results);

}  // namespace tublex

#endif  // TUBLEX_RUNTIME_HPP

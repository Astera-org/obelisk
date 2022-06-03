namespace py optimize
namespace go optimize
namespace cpp optimize

struct HyperParameter {
    1: required string name,
    2: required double center,
    3: optional double stddev,
    4: optional double min,
    5: optional double max,
    6: optional string type_of_scale, # Could be an enum, should be LINEAR or LOG
    7: optional bool is_integer,
}

struct Suggestions {
    1: string observationId,
    2: map<string, double> parameterSuggestions,
    3: map<string, string> metadata,
}

service ParameterOptimizer {

    void init(1: list<HyperParameter> parameters, 2: string runId,
        3: bool restart),

    Suggestions suggest(),

    void observe(1: string observationId, 2: double value),
}

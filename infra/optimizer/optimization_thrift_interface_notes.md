Notes about what the Thrift interface for optimizers should be.

This should be a wrapper around BONES, optuna, etc.

Initialization passes in a list of params, and for each param, a name, center, stddev, min, max, integer-or-float, and type of scaling (linear, log, etc.) 
Initialization should also pass in a run id and a boolean for whether to restart or not.

Suggest should ask the optimizer for a suggestion. It essentially just passes in nothing and gets back a string:number map. Some of the numbers will be ints, 
but that conversion can occur afterward. Maybe the return value can contain a separate string:string metadata map, 
like with confidence or something. It should also return a string id for the observation.

Observe should pass in a (string, number) pair reporting on the value observed for a string id. No return value.
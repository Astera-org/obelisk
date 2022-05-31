namespace py env
namespace go env
namespace cpp env

struct Shape {
       1: required list<i32> shape,
       2: optional list<i32> stride,
       3: optional list<string> names
}

struct ETensor {
       1: required Shape shape,
       2: required list<double> values
}

struct SpaceSpec {
       1: optional Shape shape,
       2: double min,
       3: double max,
       4: optional list<string> discreteLabels
}

struct Action {
       1: optional SpaceSpec actionShape
       2: optional ETensor vector
       3: i32 discreteOption
}

typedef map<string, SpaceSpec> Space
typedef map<string, ETensor> Observations
typedef map<string, Action> Actions

service Agent {

  map<string, string> init(1:Space actionSpace, 2:Space observationSpace),

  Actions step(1:Observations observations, 2:string debug)
}

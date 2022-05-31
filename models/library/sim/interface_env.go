package sim

import (
	"github.com/emer/emergent/env"
	"github.com/emer/etable/etensor"
)

// Environment defines expected getters and setters for our environment variables.
// Most important functions are Init, Step, State
type Environment interface {
	SetName(name string)
	Name() string
	Desc() string

	// Init initializes the world. This might involve loading x, y pairs, or creating a world.
	Init(run int)

	// Step increments the world one timestep. Possibly it should take a delta time argument.
	Step()

	// State reports what the state of inputs are for a particular layer (e.g. visual or auditory input). It also returns the correct output in a supervised learning task.
	State(layerName string) etensor.Tensor

	// TODO Add a function for reporting agent output to the world
	// AgentAction(string) // Maybe something like this?

	// TODO Decide whether these should be moved off the interface and into implementing classes.
	Order() []int
	Sequential() bool
	SetSequential(s bool)

	// Keep track of time. Unclear what these mean in the context of an embedded agent task.
	Epoch() *env.Ctr
	Trial() *env.Ctr

	Run() *env.Ctr // TODO Do not use! Pls remove!

	TrialName() *env.CurPrvString

	CurTrialName() string
	GroupName() *env.CurPrvString

	NameCol() string
	SetNameCol(s string)
	GroupCol() string
	SetGroupCol(s string)

	AssignTable(s string)

	InputAndOutputLayers() []string

	Validate() error

	// TODO Counter should be removed.
	Counter(scale env.TimeScales) (cur, prv int, chg bool)
}

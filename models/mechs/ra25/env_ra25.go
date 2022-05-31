package main

import (
	"github.com/Astera-org/models/library/sim"
	"github.com/emer/emergent/env"
	"github.com/emer/etable/etensor"
)

//EnvRa25 a wrapper for accessing the environment and it's variables
type EnvRa25 struct {
	env.FixedTable
	sim.Environment
}

func (env *EnvRa25) SetName(name string) {
	env.FixedTable.Nm = name
}
func (env *EnvRa25) Name() string {
	return env.FixedTable.Nm
}
func (env *EnvRa25) SetDesc(desc string) {
	env.FixedTable.Dsc = desc
}
func (env *EnvRa25) Desc() string {
	return env.FixedTable.Dsc
}

func (env *EnvRa25) Order() []int {
	return env.FixedTable.Order
}
func (env *EnvRa25) Sequential() bool {
	return env.FixedTable.Sequential
}
func (env *EnvRa25) SetSequential(s bool) {
	env.FixedTable.Sequential = s
}

// TODO Remove this from this interface and store runs on Sim.
func (env *EnvRa25) Run() *env.Ctr {
	return &env.FixedTable.Run
}
func (env *EnvRa25) Epoch() *env.Ctr {
	return &env.FixedTable.Epoch
}
func (env *EnvRa25) Trial() *env.Ctr {
	return &env.FixedTable.Trial
}
func (env *EnvRa25) TrialName() *env.CurPrvString {
	return &env.FixedTable.TrialName
}
func (env *EnvRa25) GroupName() *env.CurPrvString {
	return &env.FixedTable.GroupName
}
func (env *EnvRa25) NameCol() string {
	return env.FixedTable.NameCol
}
func (env *EnvRa25) SetNameCol(s string) {
	env.FixedTable.NameCol = s
}
func (env *EnvRa25) GroupCol() string {
	return env.FixedTable.GroupCol
}
func (env *EnvRa25) SetGroupCol(s string) {
	env.FixedTable.GroupCol = s
}

func (env *EnvRa25) Validate() error {
	return env.FixedTable.Validate()
}
func (env *EnvRa25) Init(run int) {
	env.FixedTable.Init(run)
}

func (env *EnvRa25) CurTrialName() string {
	return env.TrialName().Cur
}
func (env *EnvRa25) Step() {
	env.FixedTable.Step()
}

func (env *EnvRa25) State(s string) etensor.Tensor {
	return env.FixedTable.State(s)
}

func (env *EnvRa25) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
	return env.FixedTable.Counter(scale)
}

func (env *EnvRa25) InputAndOutputLayers() []string {
	return []string{"Input", "Output"}
}

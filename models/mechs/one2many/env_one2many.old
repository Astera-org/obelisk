package main

import (
	"github.com/Astera-org/models/library/sim"
	"github.com/emer/emergent/env"
	"github.com/emer/etable/etensor"
)

//EnvOne2Many a wrapper for accessing the environment and it's variables
type EnvOne2Many struct {
	env.FixedTable
	sim.Environment
}

func (env *EnvOne2Many) SetName(name string) {
	env.FixedTable.Nm = name
}
func (env *EnvOne2Many) Name() string {
	return env.FixedTable.Nm
}
func (env *EnvOne2Many) SetDesc(desc string) {
	env.FixedTable.Dsc = desc
}
func (env *EnvOne2Many) Desc() string {
	return env.FixedTable.Dsc
}

func (env *EnvOne2Many) Order() []int {
	return env.FixedTable.Order
}
func (env *EnvOne2Many) Sequential() bool {
	return env.FixedTable.Sequential
}
func (env *EnvOne2Many) SetSequential(s bool) {
	env.FixedTable.Sequential = s
}

func (env *EnvOne2Many) Run() *env.Ctr {
	return &env.FixedTable.Run
}
func (env *EnvOne2Many) Epoch() *env.Ctr {
	return &env.FixedTable.Epoch
}
func (env *EnvOne2Many) Trial() *env.Ctr {
	return &env.FixedTable.Trial
}
func (env *EnvOne2Many) TrialName() *env.CurPrvString {
	return &env.FixedTable.TrialName
}
func (env *EnvOne2Many) GroupName() *env.CurPrvString {
	return &env.FixedTable.GroupName
}
func (env *EnvOne2Many) NameCol() string {
	return env.FixedTable.NameCol
}
func (env *EnvOne2Many) SetNameCol(s string) {
	env.FixedTable.NameCol = s
}
func (env *EnvOne2Many) GroupCol() string {
	return env.FixedTable.GroupCol
}
func (env *EnvOne2Many) SetGroupCol(s string) {
	env.FixedTable.GroupCol = s
}

func (env *EnvOne2Many) Validate() error {

	return env.FixedTable.Validate()
}
func (env *EnvOne2Many) Init(run int) {
	env.FixedTable.Init(run)
}

func (env *EnvOne2Many) CurTrialName() string {
	return env.TrialName().Cur
}
func (env *EnvOne2Many) Step() {
	env.FixedTable.Step()
}

func (env *EnvOne2Many) State(s string) etensor.Tensor {
	return env.FixedTable.State(s)
}

func (env *EnvOne2Many) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
	return env.FixedTable.Counter(scale)
}

func (env *EnvOne2Many) InputAndOutputLayers() []string {
	return []string{"Input", "Output"}
}

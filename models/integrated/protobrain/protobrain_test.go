package main

import (
	log "github.com/Astera-org/easylog"
	"github.com/Astera-org/models/agent"
	"github.com/emer/axon/axon"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/looper"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/goki/gi/gi"
	"testing"
)

// ProtoworldTest ensures that basic actions should return specific outputs,
type ProtoworldTest struct {
	agent.WorldInterface
	obs     map[string]etensor.Tensor
	actions map[string]agent.Action
}

func (pworld *ProtoworldTest) New(path string) *ProtoworldTest {
	t := etable.Table{}
	t.OpenCSV(gi.FileName(path), ',')
	pworld.obs = make(map[string]etensor.Tensor)
	pworld.actions = make(map[string]agent.Action)
	for _, name := range t.ColNames {
		pworld.obs[name] = t.ColByName(name)
	}
	return pworld
}
func (pworld *ProtoworldTest) InitWorld(details map[string]string) (actionSpace map[string]agent.SpaceSpec, observationSpace map[string]agent.SpaceSpec) {
	return nil, nil
}
func (pworld *ProtoworldTest) StepWorld(actions map[string]agent.Action, agentDone bool) (done bool, debug string) {
	pworld.actions = actions
	return true, ""
}
func (pworld *ProtoworldTest) Observe(name string) etensor.Tensor {
	return pworld.obs[name]
}
func (pworld *ProtoworldTest) SetObservations(obs map[string]etensor.Tensor) {
}

// ConfigLoops configures the control loops
func (ss *Sim) ConfigTestLoop() *looper.Manager {
	manager := looper.NewManager()
	manager.AddStack(etime.Train).AddTime(etime.Run, 1).AddTime(etime.Epoch, 1).AddTime(etime.Trial, 1).AddTime(etime.Cycle, 200)
	axon.LooperStdPhases(manager, &ss.Time, ss.Net.AsAxon(), 150, 199) // plus phase timing
	plusPhase, _ := manager.GetLoop(etime.Train, etime.Cycle).EventByName("PlusPhase")
	plusPhase.OnEvent.Add("SendActionsThenStep", func() {
		agent.AgentSendActionAndStep(ss.Net.AsAxon(), ss.WorldEnv)
	})
	mode := etime.Train // For closures
	stack := manager.Stacks[mode]
	stack.Loops[etime.Trial].OnStart.Add("Observe", ss.OnObserve)
	manager.GetLoop(etime.Train, etime.Run).OnStart.Add("NewRun", ss.NewRun)
	axon.LooperSimCycleAndLearn(manager, ss.Net.AsAxon(), &ss.Time, &ss.NetDeets.ViewUpdt)
	return manager
}

//load csv datafile
func TestProto(t *testing.T) {
	log.SetLevel(900)() // suppress logging
	var sim Sim
	sim.Net = sim.ConfigNet()
	sim.Loops = sim.ConfigLoops()
	sim.WorldEnv = (&ProtoworldTest{}).New("testdata.csv")
	sim.ActionHistory = &etable.Table{} //A recording of actions taken and actions predicted
	sim.NewRun()
	sim.Loops.Run(etime.Train)
	//sim.Loops.Step(etime.Train, 1, etime.Trial)
	//sim.Loops.Step(etime.Train, 1, etime.Trial)
	print("HI")
}

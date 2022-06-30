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

// ConfigLoops configures the control loops, pass a point to record pct record over multiple time stpes
func (ss *Sim) ConfigTestLoop(pct_correct *[]float64) *looper.Manager {
	originalLoop := ss.ConfigLoops()
	originalLoop.GetLoop(etime.Train, etime.Trial).OnEnd.Add("TrialLoss", func() {
		loss := (ss.Net.LayerByName("VL").(axon.AxonLayer).AsAxon().PctUnitErr())
		(*pct_correct)[0] = loss
	})
	return originalLoop

}

//TestProto checks that basic activity actually gets sent through the network and we get NON zero values
func TestProto(t *testing.T) {
	gConfig.Load()
	testPath := gConfig.TESTFILE
	log.SetLevel(900)()          // suppress logging
	pctCorrect := []float64{0.0} //maybe we want
	var sim Sim
	sim.Net = sim.ConfigNet()
	sim.Loops = sim.ConfigTestLoop(&pctCorrect)
	pworldTest := (&ProtoworldTest{}).New(testPath)
	sim.WorldEnv = pworldTest
	sim.ActionHistory = &etable.Table{} //A recording of actions taken and actions predicted
	sim.NewRun()
	sim.Loops.Step(etime.Train, 1, etime.Trial)
	if pctCorrect[0] < 0.04 {
		t.Errorf("PctUnitErr() should be above 0.04 but got %f. This implies activity not getting sent or ground truth not compared", pctCorrect[0])
	}

	//check activity of network
	for name, _ := range pworldTest.actions {
		vector := pworldTest.actions[name].Vector
		sum := 0.0
		for i := 0; i < vector.Len(); i++ {
			sum += vector.FloatVal1D(i)
		}
		avg := sum / float64(vector.Len())
		if avg < .05 {
			t.Errorf("Action %s should have SOME avg activity but got %f", name, avg)
		}
	}
}

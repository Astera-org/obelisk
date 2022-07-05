package main

import (
	"fmt"
	"github.com/Astera-org/obelisk/models/library/sim"
	"github.com/emer/emergent/etime"
	"github.com/emer/etable/etable"
	"os"
	"testing"
)

// Commented out because the args parsing within Config can only be called once in a tests file.
//func TestConfigNet(t *testing.T) {
//	simo := sim.Sim{}
//	simo.DefineSimVariables()
//	simo.CmdArgs.NoGui = true
//	Config(&simo)
//	neto := axon.Network{}
//	ConfigNet(&simo, &neto)
//	if len(neto.Layers) != 4 {
//		t.Errorf("Expected network to be configured differently")
//	}
//}

func TestConfigPats(t *testing.T) {
	ss := sim.Sim{}
	ss.Pats = &etable.Table{}
	ConfigPats(&ss)
	if ss.Pats.Rows < 10 {
		t.Errorf("Expected more patterns than that!")
	}
}

// In GoLand, right-click the play triangle by this method and select "Profile TestModelTraining with 'CPU Profiler'"s
// TODO Modify this test to ensure that LastZero is achieved in under 100 epochs
func TestModelTraining(t *testing.T) {
	var TheSim sim.Sim
	TheSim.New()
	// TODO Add params
	os.Args = append(os.Args, "-nogui=true")
	os.Args = append(os.Args, "-runs=1")
	os.Args = append(os.Args, "-epochs=100")
	Config(&TheSim)
	TheSim.RunFromArgs()
	runlog := TheSim.Logs.Table(etime.Train, etime.Run)
	println(fmt.Sprintf("FirstZero: %.0f\tLastZero: %.0f", runlog.CellFloat("FirstZero", 0), runlog.CellFloat("LastZero", 0)))
	if runlog.CellFloat("FirstZero", 0) < 0 {
		t.Errorf("No FirstZero!")
	}
	if runlog.CellFloat("LastZero", 0) < 0 {
		t.Errorf("No LastZero!")
	}
}

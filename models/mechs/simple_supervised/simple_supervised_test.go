package main

import (
	"github.com/Astera-org/obelisk/models/library/autoui"
	"github.com/emer/emergent/etime"
	"testing"
)

// This doesn't actually test anything, but it can be used for profiling.
func TestSupervised(t *testing.T) {
	var sim Sim
	sim.WorldEnv = sim.ConfigEnv()
	sim.Net = sim.ConfigNet()
	sim.Loops = sim.ConfigLoops()
	sim.Loops.GetLoop(etime.Train, etime.Epoch).Counter.Max = 1

	userInterface := autoui.AutoUI{
		StructForView:             &sim,
		Looper:                    sim.Loops,
		Network:                   sim.Net.EmerNet,
		ViewUpdt:                  &sim.ViewUpdt,
		AppName:                   "Simple Supervised",
		AppTitle:                  "Random Associator for Supervised Task",
		AppAbout:                  `Learn to memorize random pattern pairs presented as input/output.`,
		AddNetworkLoggingCallback: autoui.AddCommonLogItemsForOutputLayers,
		RasterLayers:              []string{"Input", "Hidden1", "Hidden2", "Output"}, // Same as from ConfigNet()
		DoLogging:                 false,
		HaveGui:                   false,
	}
	userInterface.AddDefaultLogging()
	userInterface.RunWithoutGui()

}

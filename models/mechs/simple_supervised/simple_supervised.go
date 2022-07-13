// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"log"

	"github.com/Astera-org/obelisk/models/agent"
	"github.com/Astera-org/obelisk/models/library/autoui"
	"github.com/emer/axon/axon"
	"github.com/emer/axon/deep"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/looper"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/prjn"
	"github.com/emer/etable/etensor"
)

// This file demonstrates how to do supervised learning with a simple axon network and a simple task. It creates an "RA 25 Env", which stands for "Random Associator 25 (5x5)", which provides random 5x5 patterns for the network to learn.
// In addition to creating a simple environment and a simple network, it creates a looper.Manager to control the flow of time across Runs, Epochs, and Trials. It creates a GUI to control it.

var numPatterns = 25 // How many random patterns. Each pattern is one trial per epoch.

func main() {
	var sim Sim
	sim.WorldEnv = sim.ConfigEnv()
	sim.Net = sim.ConfigNet()
	sim.Loops = sim.ConfigLoops()

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
		DoLogging:                 true,
		HaveGui:                   true,
	}
	userInterface.Start() // Start blocks, so don't put any code after this.
}

// Sim encapsulates working data for the simulation model, keeping all relevant state information organized and available without having to pass everything around.
type Sim struct {
	Net      *deep.Network        `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	Params   emer.Params          `view:"inline" desc:"all parameter management"`
	ViewUpdt netview.ViewUpdt     `desc:"netview update parameters"`
	Loops    *looper.Manager      `view:"no-inline" desc:"contains looper control loops for running sim"`
	WorldEnv agent.WorldInterface `desc:"Training environment -- contains everything about iterating over input / output patterns over training"`
	Time     axon.Time            `desc:"axon timing parameters and state"`
}

func (ss *Sim) ConfigEnv() agent.WorldInterface {
	return &Ra25Env{PatternSize: 5, NumPatterns: numPatterns}
}

func (ss *Sim) ConfigNet() *deep.Network {
	// A simple network for demonstration purposes.
	net := &deep.Network{}
	net.InitName(net, "RA25")
	inp := net.AddLayer2D("Input", 5, 5, emer.Input)
	hid1 := net.AddLayer2D("Hidden1", 8, 8, emer.Hidden)
	hid2 := net.AddLayer2D("Hidden2", 8, 8, emer.Hidden)
	out := net.AddLayer2D("Output", 5, 5, emer.Target)
	full := prjn.NewFull()
	net.ConnectLayers(inp, hid1, full, emer.Forward)
	net.BidirConnectLayers(hid1, hid2, full)
	net.BidirConnectLayers(hid2, out, full)

	net.Defaults()
	ss.Params.Params = ParamSets
	ss.Params.AddNetwork(net.AsAxon())
	ss.Params.SetObject("Network")
	err := net.Build()
	if err != nil {
		log.Println(err)
		return nil
	}
	return net
}

func (ss *Sim) NewRun() {
	ss.Net.InitWts()
}

// ConfigLoops configures the control loops
func (ss *Sim) ConfigLoops() *looper.Manager {
	manager := looper.NewManager()
	manager.AddStack(etime.Train).AddTime(etime.Run, 1).AddTime(etime.Epoch, 100).AddTime(etime.Trial, numPatterns).AddTime(etime.Cycle, 200)

	axon.LooperStdPhases(manager, &ss.Time, ss.Net.AsAxon(), 150, 199) // plus phase timing

	axon.LooperSimCycleAndLearn(manager, ss.Net.AsAxon(), &ss.Time, &ss.ViewUpdt)

	plusPhase, ok := manager.GetLoop(etime.Train, etime.Cycle).EventByName("PlusPhase")
	if !ok {
		panic("PlusPhase not found")
	}
	plusPhase.OnEvent.Add("SendActionsThenStep", func() {
		// Check the action at the beginning of the Plus phase, before the teaching signal is introduced.
		agent.AgentSendActionAndStep(ss.Net.AsAxon(), ss.WorldEnv)
	})

	// Trial Stats and Apply Input
	stack := manager.Stacks[etime.Train]
	stack.Loops[etime.Trial].OnStart.Add("Observe", func() {
		agent.AgentApplyInputs(ss.Net.AsAxon(), ss.WorldEnv, "Input", func(spec agent.SpaceSpec) etensor.Tensor {
			return ss.WorldEnv.Observe("Input")
		})
		// Although ground truth output is applied here, it won't actually be clamped until PlusPhase is called, because it's a layer of type Target.
		agent.AgentApplyInputs(ss.Net.AsAxon(), ss.WorldEnv, "Output", func(spec agent.SpaceSpec) etensor.Tensor {
			return ss.WorldEnv.Observe("Output")
		})
	})

	manager.GetLoop(etime.Train, etime.Run).OnStart.Add("NewRun", ss.NewRun)
	manager.GetLoop(etime.Train, etime.Run).OnStart.Add("NewPatterns", func() { ss.WorldEnv.InitWorld(nil) })

	// Initialize and print loop structure, then add to Sim
	fmt.Println(manager.DocString())
	return manager
}

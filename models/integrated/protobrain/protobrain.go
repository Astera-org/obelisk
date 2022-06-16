// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	"github.com/emer/emergent/emer"

	"github.com/Astera-org/models/library/autoui"
	"github.com/Astera-org/worlds/network_agent"
	"github.com/emer/axon/axon"
	"github.com/emer/axon/deep"
	"github.com/emer/emergent/agent"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/looper"
	"github.com/emer/etable/etensor"
	"github.com/pkg/profile"
)

// Protobrain demonstrates a network model that has elements of cortical visual perception and a rudimentary action system.
// It is not reward motivated, and instead it learns to approximate a behavior heuristic. It is intended to be used with
// the world found in github.com/Astera-org/obelisk//worlds/integrated/fworld.

var gConfig Config

func main() {
	// note: uncomment this to get debugging on vulkan gui issues
	// vkos.VkOsDebug = true

	gConfig.Load() // LATER specify the .cfg as a cmd line arg

	if gConfig.PROFILE {
		fmt.Println("Starting profiling")
		defer profile.Start(profile.ProfilePath(".")).Stop()
	}

	var sim Sim
	sim.Net = sim.ConfigNet()
	sim.Loops = sim.ConfigLoops()
	world, serverFunc := network_agent.GetWorldAndServerFunc(sim.Loops)
	sim.WorldEnv = world

	userInterface := &autoui.AutoUI{
		StructForView:             &sim,
		Looper:                    sim.Loops,
		Network:                   sim.Net.EmerNet,
		ViewUpdt:                  &sim.NetDeets.ViewUpdt,
		AppName:                   "Protobrain solves FWorld",
		AppTitle:                  "Protobrain",
		AppAbout:                  `Learn to mimic patterns coming from a teacher signal in a flat grid world.`,
		AddNetworkLoggingCallback: autoui.AddCommonLogItemsForOutputLayers,
		DoLogging:                 true,
		HaveGui:                   gConfig.GUI,
		StartAsServer:             true,
		ServerFunc:                serverFunc,
	}
	userInterface.Start() // Start blocks, so don't put any code after this.
}

// Sim encapsulates working data for the simulation model, keeping all relevant state information organized and available without having to pass everything around.
type Sim struct {
	Net      *deep.Network        `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	NetDeets NetworkDeets         `desc:"Contains details about the network."`
	Loops    *looper.Manager      `view:"no-inline" desc:"contains looper control loops for running sim"`
	WorldEnv agent.WorldInterface `desc:"Training environment -- contains everything about iterating over input / output patterns over training"`
	Time     axon.Time            `desc:"axon timing parameters and state"`
	LoopTime string               `desc:"Printout of the current time."`
}

func (ss *Sim) ConfigNet() *deep.Network {
	net := &deep.Network{}
	DefineNetworkStructure(&ss.NetDeets, net)
	return net
}

// ConfigLoops configures the control loops
func (ss *Sim) ConfigLoops() *looper.Manager {
	manager := looper.NewManager()
	manager.AddStack(etime.Train).AddTime(etime.Run, 1).AddTime(etime.Epoch, 100).AddTime(etime.Trial, 1).AddTime(etime.Cycle, 200)

	axon.LooperStdPhases(manager, &ss.Time, ss.Net.AsAxon(), 150, 199) // plus phase timing

	plusPhase, ok := manager.GetLoop(etime.Train, etime.Cycle).EventByName("PlusPhase")
	if !ok {
		panic("PlusPhase not found")
	}
	plusPhase.OnEvent.Add("SendActionsThenStep", func() {
		axon.AgentSendActionAndStep(ss.Net.AsAxon(), ss.WorldEnv)
	})

	mode := etime.Train // For closures
	stack := manager.Stacks[mode]
	stack.Loops[etime.Trial].OnStart.Add("Observe", func() {
		for _, name := range ss.Net.LayersByClass(emer.Input.String()) { // DO NOT SUBMIT Make sure this works
			axon.AgentApplyInputs(ss.Net.AsAxon(), ss.WorldEnv, name, func(spec agent.SpaceSpec) etensor.Tensor {
				return ss.WorldEnv.Observe(name)
			})
		}

	})

	manager.GetLoop(etime.Train, etime.Run).OnStart.Add("NewRun", ss.NewRun)
	axon.LooperSimCycleAndLearn(manager, ss.Net.AsAxon(), &ss.Time, &ss.NetDeets.ViewUpdt)

	// Initialize and print loop structure, then add to Sim
	fmt.Println(manager.DocString())

	manager.GetLoop(etime.Train, etime.Trial).OnEnd.Add("QuickScore", func() {
		loss := ss.Net.LayerByName("VL").(axon.AxonLayer).AsAxon().PctUnitErr()
		s := fmt.Sprintf("%f", loss)
		fmt.Println("the pctuniterror is " + s)
	})

	return manager
}

// NewRun intializes a new run of the model, using the WorldMailbox.GetCounter(etime.Run) counter for the new run value
func (ss *Sim) NewRun() {
	run := ss.Loops.GetLoop(etime.Train, etime.Run).Counter.Cur
	ss.NetDeets.RndSeeds.Set(run)
	ss.NetDeets.PctCortex = 0
	ss.WorldEnv.InitWorld(nil)
	ss.Time.Reset()
	ss.Net.InitWts()
	ss.NetDeets.InitStats()
}

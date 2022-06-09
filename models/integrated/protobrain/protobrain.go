// What am I trying to do today?
// Connect egan to protobrain
// egan sends pixels to protobrain currently
// we need to map those pixels to the actual model layers

// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"github.com/emer/etable/etensor"

	"github.com/Astera-org/models/library/autoui"
	"github.com/Astera-org/worlds/network_agent"
	"github.com/emer/axon/axon"
	"github.com/emer/axon/deep"
	"github.com/emer/emergent/agent"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/looper"
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

	userInterface := autoui.AutoUI{
		StructForView:             &sim,
		Looper:                    sim.Loops,
		Network:                   sim.Net.EmerNet,
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
	axon.AddPlusAndMinusPhases(manager, &ss.Time, ss.Net.AsAxon())

	plusPhase := &manager.GetLoop(etime.Train, etime.Cycle).Events[1]
	plusPhase.OnEvent.Add("Sim:PlusPhase:SendActionsThenStep", func() {
		axon.SendActionAndStep(ss.Net.AsAxon(), ss.WorldEnv)
	})

	mode := etime.Train // For closures
	stack := manager.Stacks[mode]
	stack.Loops[etime.Trial].OnStart.Add("Sim:ResetState", func() {
		ss.Net.NewState()
		ss.Time.NewState(mode.String())
	})

	stack.Loops[etime.Trial].OnStart.Add("Sim:Trial:Observe", func() {
		pixels, _ := ss.WorldEnv.(*agent.AgentProxyWithWorldCache).CachedObservations["world"]
		x, y := pixels.Dims()
		// agent is located in the middle of the bottom row
		w := World{Pixels: pixels.(*etensor.Float64), AgentX: x / 2, AgentY: 0, WorldX: x, WorldY: y}
		w.Config()
		fmt.Println("Received egan world: ", w)

		obs := w.GetAllObservations()

		for name, t := range obs {
			fmt.Println("ApplyInputs name: ", name)

			axon.ApplyInputs(ss.Net.AsAxon(), ss.WorldEnv, name, func(spec agent.SpaceSpec) etensor.Tensor {
				return t
			})
		}
	})

	manager.GetLoop(etime.Train, etime.Run).OnStart.Add("Sim:NewRun", ss.NewRun)
	axon.AddDefaultLoopSimLogic(manager, &ss.Time, ss.Net.AsAxon())

	// Initialize and print loop structure, then add to Sim
	fmt.Println(manager.DocString())

	manager.GetLoop(etime.Train, etime.Trial).OnEnd.Add("Sim:Trial:QuickScore", func() {
		loss := ss.Net.LayerByName("VL").(axon.AxonLayer).AsAxon().PctUnitErr()
		s := fmt.Sprintf("%f", loss)
		fmt.Println("the pctuniterror is " + s)
	})

	// TODO: add cos similatiry here

	return manager
}

// NewRun intializes a new run of the model, using the WorldMailbox.GetCounter(etime.Run) counter for the new run value
func (ss *Sim) NewRun() {
	ss.NetDeets.PctCortex = 0
	ss.WorldEnv.InitWorld(nil)
	ss.Time.Reset()
	ss.Net.InitWts()
	ss.NetDeets.InitStats()
}

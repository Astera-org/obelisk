// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	log "github.com/Astera-org/easylog"
	"github.com/Astera-org/models/agent"
	"github.com/Astera-org/models/library/autoui"
	"github.com/Astera-org/obelisk/infra"
	"github.com/emer/axon/axon"
	"github.com/emer/axon/deep"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/looper"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/pkg/profile"
	"os"
)

// Protobrain demonstrates a network model that has elements of cortical visual perception and a rudimentary action system.
// It is not reward motivated, and instead it learns to approximate a behavior heuristic. It is intended to be used with
// the world found in github.com/Astera-org/obelisk//worlds/integrated/fworld.

var gConfig Config

func main() {
	// note: uncomment this to get debugging on vulkan gui issues
	// vkos.VkOsDebug = true

	gConfig.Load() // LATER specify the .cfg as a cmd line arg

	err := log.Init(
		log.SetLevel(log.INFO),
		log.SetFileName("brain.log"),
	)
	if err != nil {
		panic(err)
	}

	log.Info("Starting Protobrain ==========")

	if gConfig.PROFILE {
		log.Info("Starting profiling")
		defer profile.Start(profile.ProfilePath(".")).Stop()
	}
	var sim Sim
	sim.Net = sim.ConfigNet()
	sim.Loops = sim.ConfigLoops()
	sim.WorldEnv = &agent.NetworkWorld{}
	sim.ActionHistory = &etable.Table{} //A recording of actions taken and actions predicted

	if gConfig.WORKER {
		sim.startWorkerLoop()
	} else {
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
			ServerFunc:                sim.startWorkerLoop,
		}
		userInterface.Start() // Start blocks, so don't put any code after this.
	}

}

// Sim encapsulates working data for the simulation model, keeping all relevant state information organized and available without having to pass everything around.
type Sim struct {
	Net           *deep.Network        `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	NetDeets      NetworkDeets         `desc:"Contains details about the network."`
	Loops         *looper.Manager      `view:"no-inline" desc:"contains looper control loops for running sim"`
	WorldEnv      agent.WorldInterface `desc:"Training environment -- contains everything about iterating over input / output patterns over training"`
	Time          axon.Time            `desc:"axon timing parameters and state"`
	LoopTime      string               `desc:"Printout of the current time."`
	NumSteps      int32
	ActionHistory *etable.Table `desc:"a recording of actions taken and predicted per trial"`
}

func (ss *Sim) ConfigNet() *deep.Network {
	net := &deep.Network{}
	DefineNetworkStructure(&ss.NetDeets, net)
	return net
}

func wrapperTensorFloat(value float64) etensor.Tensor {
	predictedTensor := etensor.New(etensor.FLOAT64, []int{1.0}, nil, nil)
	predictedTensor.SetFloat1D(0, float64(value))
	return predictedTensor
}
func wrapperTensorString(value string) etensor.Tensor {
	predictedTensor := etensor.New(etensor.STRING, []int{1}, nil, nil)
	predictedTensor.SetString1D(0, value)

	return predictedTensor
}

func createActionHistoryRow(predicted, groundtruth, run float64, timescale string) etable.Table {
	table := etable.Table{}
	table.AddCol(wrapperTensorFloat(predicted), "Predicted")
	table.AddCol(wrapperTensorFloat(groundtruth), "GroundTruth")
	table.AddCol(wrapperTensorString(timescale), "Timescale")
	table.AddCol(wrapperTensorFloat(run), "Run")
	table.SetNumRows(1)
	return table

}

func (ss *Sim) AddActionHistory(observations map[string]etensor.Tensor, timeScale string) {
	bestAction, actionExists := observations["Heuristic"]
	prevPredictedAction, predictedExists := observations["PredictedActionLastTimeStep"]
	if (actionExists == false) || (predictedExists == false) { //if not getting info across network, perhaps using diff world
		log.Error("Heuristic or Predicted Action keys are not found in observations, cannot log results")
	} else {
		floatAction := float64(bestAction.FloatVal1D(0))
		floatPredictedPrevious := float64(prevPredictedAction.FloatVal1D(0))
		runNum := float64(ss.Loops.GetLoop(etime.Train, etime.Run).Counter.Cur)
		dframe := createActionHistoryRow(-1.0, floatAction, runNum, timeScale)
		if ss.ActionHistory.Rows == 0 {
			ss.ActionHistory = &dframe
		} else {
			ss.ActionHistory.AppendRows(&dframe)
			ss.ActionHistory.SetCellFloat("Predicted", ss.ActionHistory.Rows-2, floatPredictedPrevious) //set n-1 one values
		}
	}
}

// ConfigLoops configures the control loops
func (ss *Sim) ConfigLoops() *looper.Manager {
	manager := looper.NewManager()
	manager.AddStack(etime.Train).AddTime(etime.Run, 1).AddTime(etime.Epoch, 100).AddTime(etime.Trial, 1).AddTime(etime.Cycle, 200)

	axon.LooperStdPhases(manager, &ss.Time, ss.Net.AsAxon(), 150, 199) // plus phase timing

	plusPhase, ok := manager.GetLoop(etime.Train, etime.Cycle).EventByName("PlusPhase")
	if !ok {
		log.Fatal("PlusPhase not found")
	}
	// TODO this doesn't make sense. we should wait for the next Step call to come in from the
	plusPhase.OnEvent.Add("SendActionsThenStep", func() {
		agent.AgentSendActionAndStep(ss.Net.AsAxon(), ss.WorldEnv)
	})

	mode := etime.Train // For closures
	stack := manager.Stacks[mode]
	stack.Loops[etime.Trial].OnStart.Add("Observe", ss.OnObserve)

	manager.GetLoop(etime.Train, etime.Run).OnStart.Add("NewRun", ss.NewRun)
	axon.LooperSimCycleAndLearn(manager, ss.Net.AsAxon(), &ss.Time, &ss.NetDeets.ViewUpdt)

	// Initialize and print loop structure, then add to Sim
	log.Info(manager.DocString())

	manager.GetLoop(etime.Train, etime.Trial).OnEnd.Add("QuickScore", func() {
		loss := ss.Net.LayerByName("VL").(axon.AxonLayer).AsAxon().PctUnitErr()
		s := fmt.Sprintf("%f", loss)
		log.Info("the pctuniterror is " + s)
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

func (sim *Sim) OnObserve() {
	//set inputs to all layers of type input
	for _, name := range sim.Net.LayersByClass(emer.Input.String()) {
		agent.AgentApplyInputs(sim.Net.AsAxon(), sim.WorldEnv, name)
	}
	//set expected output/groundtruth to layers of type target
	for _, name := range sim.Net.LayersByClass(emer.Target.String()) {
		agent.AgentApplyInputs(sim.Net.AsAxon(), sim.WorldEnv, name)
	}
}

func (sim *Sim) OnStep(obs map[string]etensor.Tensor) map[string]agent.Action {
	sim.NumSteps++
	if sim.NumSteps >= gConfig.LIFETIME {
		// TODO figure score and seconds
		log.Info("LIFETIME reached")
		sim.ActionHistory.SaveCSV("results.csv", ',', true)
		infra.WriteResults(.5, sim.NumSteps, 100)
		os.Exit(0)
	}

	sim.WorldEnv.SetObservations(obs)

	log.Info("OnStep: ", sim.NumSteps)
	sim.Loops.Step(sim.Loops.Mode, 1, etime.Trial)

	sim.AddActionHistory(obs, etime.Trial.String()) //record history as discrete values

	actions := agent.GetAction(sim.Net.AsAxon())

	return actions
}

func (sim *Sim) startWorkerLoop() {
	// start the server listening for the world telling you to step
	// 		Step Called by the world:
	// 		increment the count of steps
	// 		run the lopper
	// 		get the action from the current brain state
	// 		return the action to the world
	// write results after LIFETIME steps
	// exit
	addr := fmt.Sprint("127.0.0.1:", gConfig.INTERNAL_PORT)
	agent.StartServer(addr, sim.OnStep)
}

package main

import (
	"context"
	"fmt"
	"github.com/Astera-org/worlds/network"
	"github.com/Astera-org/worlds/network/gengo/env"
	"github.com/Astera-org/worlds/network_agent"
	"github.com/emer/emergent/agent"
	"github.com/emer/emergent/patgen"
	"github.com/emer/etable/etensor"
)

type ExampleWorld struct {
	netAttributes    NetCharacteristics
	observationShape map[string][]int
	observations     map[string]*etensor.Float32
}

// Observe Returns a tensor for the named modality. E.g. “x” or “vision” or “reward”
func (world *ExampleWorld) Observe(name string) etensor.Tensor {
	if name == "VL" { //if type target
		return world.observations[name]
	} else { //if an input
		if world.observations[name] == nil {
			spaceSpec := world.netAttributes.ObservationMapping[name]
			world.observations[name] = etensor.NewFloat32(spaceSpec.ContinuousShape, spaceSpec.Stride, nil)
			patgen.PermutedBinaryRows(world.observations[name], 1, 1, 0)
		}

	}
	return world.observations[name]
}

// StepWorld steps the index of the current pattern.
func (world *ExampleWorld) StepWorld(actions map[string]agent.Action, agentDone bool) (worldDone bool, debug string) {
	return false, ""
}

// InitWorld Initializes or reinitialize the world, todo, change from being hardcoded for emery
func (world *ExampleWorld) InitWorld(details map[string]string) (actionSpace map[string]agent.SpaceSpec, observationSpace map[string]agent.SpaceSpec) {

	world.netAttributes = (&NetCharacteristics{}).Init()
	world.observations = make(map[string]*etensor.Float32)
	world.observations["VL"] = etensor.NewFloat32(world.netAttributes.ActionMapping["VL"].ContinuousShape, world.netAttributes.ActionMapping["VL"].Stride, nil)

	patgen.PermutedBinaryRows(world.observations["VL"], 1, 1, 0)

	return map[string]agent.SpaceSpec{"VL": world.netAttributes.ObservationMapping["VL"]}, world.netAttributes.ObservationMapping
}

func (world *ExampleWorld) getAllObservations() map[string]*env.ETensor {
	tensorMap := make(map[string]*env.ETensor)

	for name, _ := range world.netAttributes.ObservationMapping {
		tensor := network_agent.FromTensor(world.Observe(name))
		tensorMap[name] = tensor
	}
	return tensorMap
}

func main() {
	bestWorld := ExampleWorld{}
	bestWorld.InitWorld(nil)
	agent := network.MakeClient()
	var defaultCtx = context.Background()

	//shape := env.Shape{bestWorld.netAttributes.ActionMapping["VL"].ContinuousShape}
	//actionSpaceSpec := env.SpaceSpec{bestWorld.netAttributes.ActionMapping["VL"].ContinuousShape}

	agent.Init(defaultCtx, nil, nil)
	fmt.Println("Called INIT")
	for {
		observations := bestWorld.getAllObservations()
		fmt.Println("about to do a step")
		agent.Step(defaultCtx, observations, "")
	}
}

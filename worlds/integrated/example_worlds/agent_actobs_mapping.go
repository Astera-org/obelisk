package main

import "github.com/emer/emergent/agent"

// Proof of concept for replacing the environment with a simpler environment that uses the network.

type ObservationInput struct {
	Name        string
	InputShape  []int
	StrideShape []int
	IsTarget    bool
}

func (inputShape *ObservationInput) ToSpaceSpec() agent.SpaceSpec {
	return agent.SpaceSpec{
		ContinuousShape: inputShape.InputShape,
		Stride:          inputShape.StrideShape,
		Min:             0,
		Max:             1,
	}
}

type NetCharacteristics struct {
	V2Wd ObservationInput
	V2Fd ObservationInput
	V1F  ObservationInput
	S1S  ObservationInput
	S1V  ObservationInput
	Ins  ObservationInput
	VL   ObservationInput
	Act  ObservationInput

	ObservationMapping map[string]agent.SpaceSpec
	ActionMapping      map[string]agent.SpaceSpec
}

func (netAttributes *NetCharacteristics) Init() NetCharacteristics {
	netAttributes.V2Wd = ObservationInput{"V2Wd", []int{8, 13, 4, 1}, []int{52, 4, 1, 1}, false}
	netAttributes.V2Fd = ObservationInput{"V2Fd", []int{8, 3, 4, 1}, []int{12, 4, 1, 1}, false}
	netAttributes.V1F = ObservationInput{"V1F", []int{1, 3, 5, 5}, []int{75, 25, 5, 1}, false}
	netAttributes.S1S = ObservationInput{"S1S", []int{1, 4, 2, 1}, []int{8, 2, 1, 1}, false}
	netAttributes.S1V = ObservationInput{"S1V", []int{1, 2, 16, 1}, []int{32, 16, 1, 1}, false}
	netAttributes.Ins = ObservationInput{"Ins", []int{1, 5, 16, 1}, []int{80, 16, 1, 1}, false}
	netAttributes.VL = ObservationInput{"VL", []int{5, 5}, []int{5, 1}, false}
	netAttributes.Act = ObservationInput{"Act", []int{5, 5}, []int{5, 1}, true}

	observations := []ObservationInput{netAttributes.V2Wd, netAttributes.V2Fd, netAttributes.V1F, netAttributes.S1S, netAttributes.S1V, netAttributes.Ins, netAttributes.Act}
	actions := []ObservationInput{netAttributes.VL}

	netAttributes.ObservationMapping = make(map[string]agent.SpaceSpec)
	for _, ob := range observations {
		netAttributes.ObservationMapping[ob.Name] = ob.ToSpaceSpec()
	}

	netAttributes.ActionMapping = make(map[string]agent.SpaceSpec)
	for _, action := range actions {
		netAttributes.ActionMapping[action.Name] = action.ToSpaceSpec()
	}

	return *netAttributes
}

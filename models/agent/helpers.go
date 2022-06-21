package agent

import (
	"fmt"

	"github.com/emer/axon/axon"
	"github.com/emer/emergent/emer"
	"github.com/emer/etable/etensor"
)

/////////////////////////////////////////////////////
// Agent

// AgentSendActionAndStep takes action for this step, using either decoded cortical
// or reflexive subcortical action from env.
func AgentSendActionAndStep(net *axon.Network, ev WorldInterface) {
	// Iterate over all Target (output) layers
	actions := map[string]Action{}
	for _, lnm := range net.LayersByClass(emer.Target.String()) {
		ly := net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		vt := &etensor.Float32{}      // TODO Maybe make this more efficient by holding a copy of the right size?
		ly.UnitValsTensor(vt, "ActM") // ActM is neuron activity
		actions[lnm] = Action{Vector: vt, ActionShape: &SpaceSpec{
			ContinuousShape: vt.Shp,
			Stride:          vt.Strd,
			Min:             0,
			Max:             1,
		}}
	}
	_, debug := ev.StepWorld(actions, false)
	if debug != "" {
		fmt.Println("Got debug from Step: " + debug)
	}
}

// AgentApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func AgentApplyInputs(net *axon.Network, en WorldInterface, layerName string, patfunc func(spec SpaceSpec) etensor.Tensor) {
	lyi := net.LayerByName(layerName)
	if lyi == nil {
		fmt.Printf("layer not found: %s\n", layerName)
		return
	}
	lyi.(axon.AxonLayer).InitExt() // Clear any existing inputs
	ly := lyi.(axon.AxonLayer).AsAxon()
	ss := SpaceSpec{ContinuousShape: lyi.Shape().Shp, Stride: lyi.Shape().Strd}
	pats := patfunc(ss)
	if pats != nil {
		ly.ApplyExt(pats)
	}
}

package agent

import (
	"github.com/emer/axon/axon"
	"github.com/emer/emergent/emer"
	"github.com/emer/etable/etensor"
	log "github.com/zajann/easylog"
)

/////////////////////////////////////////////////////
// Agent

func GetAction(net *axon.Network) map[string]Action {
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
	return actions
}

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
		log.Info("Got debug from Step: " + debug)
	}
}

// AgentApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func AgentApplyInputs(net *axon.Network, world WorldInterface, layerName string) {
	lyi := net.LayerByName(layerName)
	if lyi == nil {
		log.Error("layer not found: %s\n", layerName)
		return
	}
	lyi.(axon.AxonLayer).InitExt() // Clear any existing inputs
	ly := lyi.(axon.AxonLayer).AsAxon()

	pats := world.Observe(layerName)
	if pats != nil {
		ly.ApplyExt(pats)
	}
}

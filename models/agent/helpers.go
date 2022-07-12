package agent

import (
	log "github.com/Astera-org/easylog"
	"github.com/Astera-org/obelisk/models/library/helpers"
	"github.com/emer/axon/axon"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/looper"
	"github.com/emer/etable/etensor"
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

// GetRunInfo formats the run info to send over network about num runs, steps, epochs
func GetRunInfo(manager *looper.Manager, mode etime.Modes, times ...etime.Times) map[string]Action {
	nameHash := make(map[etime.Times]string)
	nameHash[etime.Trial] = "Trial"
	nameHash[etime.Run] = "Run"
	nameHash[etime.Epoch] = "Epoch"

	infoMap := make(map[string]Action)

	for _, time := range times {
		loop := manager.GetLoop(mode, time)
		amount := loop.Counter.Cur
		max := loop.Counter.Max
		amountTensor := helpers.WrapperTensorFloat(float64(amount))
		maxTensor := helpers.WrapperTensorFloat(float64(max))
		name := nameHash[time]
		nameMax := "Max" + name

		amountAction := Action{Vector: amountTensor, ActionShape: &SpaceSpec{
			ContinuousShape: []int{1},
			Stride:          nil,
			Min:             0,
			Max:             1,
		}}

		maxAction := Action{Vector: maxTensor, ActionShape: &SpaceSpec{
			ContinuousShape: []int{1},
			Stride:          nil,
			Min:             0,
			Max:             1,
		}}
		infoMap[name] = amountAction
		infoMap[nameMax] = maxAction
	}
	return infoMap
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

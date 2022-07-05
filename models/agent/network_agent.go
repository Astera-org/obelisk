package agent

import (
	"context"

	"github.com/Astera-org/obelisk/worlds/network/gengo/env"
	thrift "github.com/apache/thrift/lib/go/thrift"
	"github.com/emer/etable/etensor"
)

type NetworkAgent struct {
	OnStep func(map[string]etensor.Tensor) map[string]Action
}

// implements the WorldInterface
type NetworkWorld struct {
	CachedObservations map[string]etensor.Tensor `desc:"Observations from the last step."`
}

func StartServer(addr string, onStep func(map[string]etensor.Tensor) map[string]Action) {
	handler := &NetworkAgent{}
	handler.OnStep = onStep
	server := MakeServer(handler, addr)
	server.Serve()
}

func MakeServer(handler env.Agent, addr string) *thrift.TSimpleServer {
	transportFactory := thrift.NewTBufferedTransportFactory(8192)
	transport, _ := thrift.NewTServerSocket(addr)
	processor := env.NewAgentProcessor(handler)
	protocolFactory := thrift.NewTBinaryProtocolFactoryConf(nil)
	server := thrift.NewTSimpleServer4(processor, transport, transportFactory, protocolFactory)
	return server
}

func (handler *NetworkAgent) Init(ctx context.Context, actionSpace env.Space,
	observationSpace env.Space) (map[string]string, error) {
	return map[string]string{}, nil
}

func (handler *NetworkAgent) Step(ctx context.Context, observations env.Observations, debug string) (env.Actions, error) {

	obs := transformObservations(observations)
	actions := handler.OnStep(obs)
	return transformActions(actions), nil
}

func (world *NetworkWorld) InitWorld(details map[string]string) (map[string]SpaceSpec, map[string]SpaceSpec) {
	// This does nothing. The external world initializes itself.
	return nil, nil
}

func (world *NetworkWorld) SetObservations(observations map[string]etensor.Tensor) {
	world.CachedObservations = observations
}

func (world *NetworkWorld) StepWorld(actions map[string]Action, agentDone bool) (bool, string) {
	// not used
	return false, ""
}

func (world *NetworkWorld) Observe(name string) etensor.Tensor {
	obs, ok := world.CachedObservations[name]
	if ok {
		return obs
	}
	return nil
}

////// helper functions

func transformActions(actions map[string]Action) env.Actions {
	if actions == nil {
		return nil
	}

	res := make(env.Actions)
	for k, v := range actions {
		res[k] = toEnvAction(&v)
	}
	return res
}

func toEnvAction(action *Action) *env.Action {
	return &env.Action{
		ActionShape:    fromSpaceSpec(action.ActionShape),
		Vector:         FromTensor(action.Vector),
		DiscreteOption: int32(action.DiscreteOption),
	}
}

func fromSpaceSpec(spec *SpaceSpec) *env.SpaceSpec {
	return &env.SpaceSpec{
		Shape: &env.Shape{Shape: toInt32(spec.ContinuousShape), Stride: toInt32(spec.Stride)},
		Min:   spec.Min,
		Max:   spec.Max,
	}
}

func FromTensor(tensor etensor.Tensor) *env.ETensor {
	res := &env.ETensor{
		Shape:  fromShape(tensor.ShapeObj()),
		Values: nil, // gets set in the next line
	}
	tensor.Floats(&res.Values)
	return res
}

func fromShape(shape *etensor.Shape) *env.Shape {
	return &env.Shape{
		Shape:  toInt32(shape.Shp),
		Stride: toInt32(shape.Strd),
		Names:  shape.Nms,
	}
}

func toInt32(xs []int) []int32 {
	if xs == nil {
		return nil
	}
	res := make([]int32, len(xs))
	for i := range xs {
		res[i] = int32(xs[i])
	}
	return res
}

func transformObservations(observations env.Observations) map[string]etensor.Tensor {
	res := make(map[string]etensor.Tensor)
	for k, v := range observations {
		res[k] = toTensor(v)
	}
	return res
}

func toTensor(envtensor *env.ETensor) etensor.Tensor {
	return etensor.NewFloat64Shape(toShape(envtensor.Shape), envtensor.Values)
}

func toShape(shape *env.Shape) *etensor.Shape {
	return etensor.NewShape(toInt(shape.Shape), toInt(shape.Stride), shape.Names)
}

func toInt(xs []int32) []int {
	if xs == nil {
		return nil
	}
	res := make([]int, len(xs))
	for i := range xs {
		res[i] = int(xs[i])
	}
	return res
}

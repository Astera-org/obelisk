package worlds

import (
	"context"
	"fmt"

	"github.com/Astera-org/worlds/network"
	"github.com/Astera-org/worlds/network/gengo/env"
)

const ADDR = "localhost:9090"

type AgentHandler struct {
	env.Agent
}

func (agent AgentHandler) Init(ctx context.Context, actionSpace env.Space,
	observationSpace env.Space) (map[string]string, error) {
	return map[string]string{"name": "GoAgentZero"}, nil
}

func (agent AgentHandler) Step(ctx context.Context, observations env.Observations, debug string) (env.Actions, error) {
	fmt.Println("Step called observations:", observations, "debug:", debug)
	action := env.Action{DiscreteOption: 0}
	return env.Actions{"move": &action}, nil
}

func main() {
	handler := AgentHandler{}
	server := network.MakeServer(handler)
	fmt.Println("listening on", ADDR)
	server.Serve()
}

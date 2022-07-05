package worlds

import (
	"context"
	"fmt"
	"github.com/hidez8891/shm"

	"github.com/Astera-org/obelisk/worlds/network"
	"github.com/Astera-org/obelisk/worlds/network/gengo/env"
)

const _ADDR = "localhost:9090"
const _AGENT_NAME = "GoAgentZero"
const _SHM_NAME = "shm_ " + _AGENT_NAME

type _AgentHandler struct {
	env.Agent
}

func (agent _AgentHandler) Init(ctx context.Context, actionSpace env.Space,
	observationSpace env.Space) (map[string]string, error) {

	// TODO: size should be a function of the observation space
	MakeShmem(_SHM_NAME, 256)

	res := map[string]string{
		"name": _AGENT_NAME,
		"shm":  _SHM_NAME}

	return res, nil
}

func (agent _AgentHandler) Step(ctx context.Context, observations env.Observations, debug string) (env.Actions, error) {
	fmt.Println("Step called observations:", observations, "debug:", debug)
	fmt.Println("agent: ", agent)
	// in practice we would reuse this buffer so we don't reallocate on every call
	rbuf := make([]byte, 256)

	// instead of reopening the shared mem every time we can remmeber it in AgentHandler
	// we would have to make the interface take a pointer to AgentHandler if we do that though, which is fine

	// here size should probably be the product of the dimensions of the tensor
	m, _ := shm.Open(_SHM_NAME, 256)
	m.Read(rbuf)
	fmt.Println("Read from shared mem: ", rbuf)

	action := env.Action{DiscreteOption: 0}
	return env.Actions{"move": &action}, nil
}

// Creates a shared memory block
func MakeShmem(name string, size int32) *shm.Memory {
	// TODO: check for errors
	m, e := shm.Create(name, size)
	fmt.Println("created shared memory: ", m, e)
	return m
}

/*
   An example of an agent that receives observations from the env using shared memory.
   The rest of the protocol is still using thrift.
   The benefit of using shared memory is less copying, so should have higher bandwidth (faster).
   Of course we have to be careful how we use that memory (synchronization).

   Some open questions: which side should create the shared mem block?
   How do we ensure the memory is not accessed at the same time?
*/
func main() {
	handler := _AgentHandler{}
	server := network.MakeServer(handler)
	fmt.Println("listening on", _ADDR)
	server.Serve()
}

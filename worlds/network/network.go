package network

import (
	"github.com/apache/thrift/lib/go/thrift"

	"github.com/Astera-org/obelisk/worlds/network/gengo/env"
)

const ADDR = "localhost:9090"

func MakeServer(handler env.Agent) *thrift.TSimpleServer {
	transportFactory := thrift.NewTBufferedTransportFactory(8192)
	transport, _ := thrift.NewTServerSocket(ADDR)
	processor := env.NewAgentProcessor(handler)
	protocolFactory := thrift.NewTBinaryProtocolFactoryConf(nil)
	server := thrift.NewTSimpleServer4(processor, transport, transportFactory, protocolFactory)
	return server
}

func MakeClient() *env.AgentClient {
	transportFactory := thrift.NewTBufferedTransportFactory(8192)
	transportSocket := thrift.NewTSocketConf(ADDR, nil)
	transport, _ := transportFactory.GetTransport(transportSocket)

	protocolFactory := thrift.NewTBinaryProtocolFactoryConf(nil)

	iprot := protocolFactory.GetProtocol(transport)
	oprot := protocolFactory.GetProtocol(transport)

	transport.Open()

	return env.NewAgentClient(thrift.NewTStandardClient(iprot, oprot))
}

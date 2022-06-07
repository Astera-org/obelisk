package main

import (
	"fmt"

	"github.com/Astera-org/obelisk/infra/gengo/infra"
	"github.com/apache/thrift/lib/go/thrift"
)

var gConfig Config

/*
- Connect to DB
- Create server
- Handle requests
*/

func main() {
	gConfig.Load()

	fmt.Println("listening on", gConfig.SERVER_ADDR)
	handler := RequestHandler{}

	server := MakeServer(handler)

	server.Serve()

}

func MakeServer(handler infra.JobCzar) *thrift.TSimpleServer {
	transportFactory := thrift.NewTBufferedTransportFactory(8192)
	transport, _ := thrift.NewTServerSocket(gConfig.SERVER_ADDR)
	processor := infra.NewJobCzarProcessor(handler)
	protocolFactory := thrift.NewTBinaryProtocolFactoryConf(nil)
	server := thrift.NewTSimpleServer4(processor, transport, transportFactory, protocolFactory)
	return server
}

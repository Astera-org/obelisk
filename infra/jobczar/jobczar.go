package main

import (
	"fmt"

	"github.com/Astera-org/obelisk/infra/gengo/infra"
	"github.com/apache/thrift/lib/go/thrift"
)

var gConfig Config

var gDatabase Database

/*
- Connect to DB
- Create server
- Handle requests
*/

func main() {
	gConfig.Load()
	gDatabase.Connect()

	fmt.Println("listening on", gConfig.SERVER_ADDR)
	handler := RequestHandler{}

	server := MakeServer(handler)

	server.Serve()
}

func MakeServer(handler infra.JobCzar) *thrift.TSimpleServer {
	transportFactory := thrift.NewTTransportFactory()
	transport, _ := thrift.NewTServerSocketTimeout(gConfig.SERVER_ADDR, 5)
	processor := infra.NewJobCzarProcessor(handler)
	protocolFactory := thrift.NewTSimpleJSONProtocolFactoryConf(nil)
	//protocolFactory := thrift.NewTBinaryProtocolFactoryConf(nil)
	server := thrift.NewTSimpleServer4(processor, transport, transportFactory, protocolFactory)
	return server
}

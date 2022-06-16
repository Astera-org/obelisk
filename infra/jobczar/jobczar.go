package main

import (
	"context"
	"fmt"
	"net/http"

	"github.com/Astera-org/obelisk/infra/gengo/infra"
	"github.com/apache/thrift/lib/go/thrift"
	"github.com/rs/cors"
	"github.com/zenazn/goji"
)

var gConfig Config
var defaultCtx = context.Background()
var gDatabase Database

/*
- Connect to DB
- Create server
- Handle requests
*/

// NewThriftHandlerFunc is a function that create a ready to use Apache Thrift Handler function
func NewThriftHandlerFunc(processor thrift.TProcessor,
	inPfactory, outPfactory thrift.TProtocolFactory) func(w http.ResponseWriter, r *http.Request) {

	return func(w http.ResponseWriter, r *http.Request) {
		transport := thrift.NewStreamTransport(r.Body, w)
		processor.Process(defaultCtx, inPfactory.GetProtocol(transport), outPfactory.GetProtocol(transport))
	}
}

func main() {
	gConfig.Load()
	gDatabase.Connect()

	fmt.Println("listening on", gConfig.SERVER_ADDR)

	// this is just a hack for localhost testing
	c := cors.New(cors.Options{
		AllowedOrigins: []string{"http://test.com", "127.0.0.1", "localhost", "localhost:8000", "localhost:9009", "null"},
	})
	goji.Use(c.Handler)

	handler := RequestHandler{}
	server := MakeServer(handler)

	go httpServer(&handler)

	server.Serve()
}

func httpServer(handler *RequestHandler) {
	processor := infra.NewJobCzarProcessor(handler)
	factory := thrift.NewTJSONProtocolFactory()

	goji.Post("/jobczar", NewThriftHandlerFunc(processor, factory, factory))

	goji.Serve()
}

func MakeServer(handler infra.JobCzar) *thrift.TSimpleServer {
	transportFactory := thrift.NewTBufferedTransportFactory(8192)
	transport, _ := thrift.NewTServerSocket(gConfig.SERVER_ADDR)
	processor := infra.NewJobCzarProcessor(handler)
	protocolFactory := thrift.NewTBinaryProtocolFactoryConf(nil)
	server := thrift.NewTSimpleServer4(processor, transport, transportFactory, protocolFactory)
	return server
}

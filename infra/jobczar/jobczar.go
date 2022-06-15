package main

import (
	"context"
	"fmt"
	"github.com/rs/cors"
	"net/http"
	"net/http/httputil"

	"github.com/Astera-org/obelisk/infra/gengo/infra"
	"github.com/apache/thrift/lib/go/thrift"
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

	// this is just a hack for localhost testing
	c := cors.New(cors.Options{
		AllowedOrigins: []string{"http://test.com", "127.0.0.1", "localhost", "localhost:8000", "localhost:9009", "null"},
	})
	goji.Use(c.Handler)

	handler := RequestHandler{}
	processor := infra.NewJobCzarProcessor(handler)
	factory := thrift.NewTJSONProtocolFactory()

	goji.Post("/jobczar", NewThriftHandlerFunc(processor, factory, factory))

	goji.Serve()
}

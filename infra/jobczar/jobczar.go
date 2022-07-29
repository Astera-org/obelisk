package main

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"os"
	"time"

	log "github.com/Astera-org/easylog"

	"github.com/Astera-org/obelisk/infra/common"
	"github.com/Astera-org/obelisk/infra/gengo/infra"
	"github.com/apache/thrift/lib/go/thrift"
	"github.com/rs/cors"
	"github.com/zenazn/goji"
)

var gConfig Config
var defaultCtx = context.Background()
var gDatabase Database
var VERSION string = "v0.1.0"

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

	err := log.Init(
		log.SetLevel(log.INFO),
		log.SetFileName("jobczar.log"),
	)
	if err != nil {
		panic(err)
	}

	gDatabase.Connect()

	// TODO: enable this once we run an actual web server for jobboard
	//if gConfig.IS_LOCALHOST {
	// this is just a hack for localhost testing
	c := cors.New(cors.Options{
		AllowedOrigins: []string{"*"},
	})
	goji.Use(c.Handler)
	//}

	handler := RequestHandler{}
	thriftServer := MakeThriftServer(handler)

	log.Info("thrift server listening on ", gConfig.THRIFT_PORT)
	log.Info("http server listening on ", gConfig.HTTP_PORT)

	go httpServer(&handler)
	go thriftServer.Serve()

	common.SignalHandler()
	inputHandler()
}

func inputHandler() {
	log.Info("Listening for input")
	for true {
		var command string
		fmt.Scan(&command)
		if len(command) == 0 {
			// this happens when we try to run the process in the background
			time.Sleep(10 * time.Second)
			continue
		}
		switch command {
		case "q":
			os.Exit(0)
		case "s":
			printStats()
		case "v":
			fmt.Println("Version: ", VERSION)
		default:
			fmt.Println("Unknown key", command, len(command))
			printHelp()
		}
	}
}

func printHelp() {
	fmt.Println("Valid Commands:")
	fmt.Println("q: quit")
	fmt.Println("s: print stats")
	fmt.Println("v: print version")
}

func printStats() {
	fmt.Println("Stats:")
	fmt.Println(" in Q:", gDatabase.GetJobCount(0))
	fmt.Println(" working:", gDatabase.GetJobCount(1))
}

func httpServer(handler *RequestHandler) {
	processor := infra.NewJobCzarProcessor(handler)
	factory := thrift.NewTJSONProtocolFactory()

	goji.Post("/jobczar", NewThriftHandlerFunc(processor, factory, factory))

	listener, err := net.Listen("tcp", fmt.Sprint(":", gConfig.HTTP_PORT))
	if err != nil {
		log.Fatal(err)
	}
	goji.ServeListener(listener)
}

func MakeThriftServer(handler infra.JobCzar) *thrift.TSimpleServer {
	transportFactory := thrift.NewTBufferedTransportFactory(8192)
	transport, _ := thrift.NewTServerSocket(fmt.Sprint(":", gConfig.THRIFT_PORT))
	processor := infra.NewJobCzarProcessor(handler)
	protocolFactory := thrift.NewTBinaryProtocolFactoryConf(nil)
	server := thrift.NewTSimpleServer4(processor, transport, transportFactory, protocolFactory)
	return server
}

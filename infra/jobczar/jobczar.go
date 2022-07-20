package main

import (
	"context"
	"fmt"
	log "github.com/Astera-org/easylog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

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

	log.Info("thrift server listening on ", gConfig.THRIFT_SERVER_ADDR)
	log.Info("http server listening on ", gConfig.HTTP_SERVER_ADDR)

	go httpServer(&handler)
	go thriftServer.Serve()

	signalHandler()
	inputHandler()
}

func signalHandler() {
	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)
	signal.Notify(sigs)
	// this happens when running in background and the stdin closes
	signal.Ignore(syscall.SIGURG, syscall.SIGTTIN)

	go func() {
		for sig := range sigs {
			log.Info("signalHandler received signal: ", sig)
			// the reason we need to custom handle this is goji intercepts it
			// but doesn't stop the entire process since we have multiple goroutines
			if sig == syscall.SIGINT || sig == syscall.SIGTERM {
				log.Info("exiting")
				os.Exit(0)
			}
		}
	}()
}

func inputHandler() {
	log.Info("Listening for input")
	for true {
		var command string
		fmt.Scan(&command)
		if len(command) == 0 {
			// this happens when we try to run the process in the background
			time.Sleep(time.Second)
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

	listener, err := net.Listen("tcp", gConfig.HTTP_SERVER_ADDR)
	if err != nil {
		log.Fatal(err)
	}
	goji.ServeListener(listener)
}

func MakeThriftServer(handler infra.JobCzar) *thrift.TSimpleServer {
	transportFactory := thrift.NewTBufferedTransportFactory(8192)
	transport, _ := thrift.NewTServerSocket(gConfig.THRIFT_SERVER_ADDR)
	processor := infra.NewJobCzarProcessor(handler)
	protocolFactory := thrift.NewTBinaryProtocolFactoryConf(nil)
	server := thrift.NewTSimpleServer4(processor, transport, transportFactory, protocolFactory)
	return server
}

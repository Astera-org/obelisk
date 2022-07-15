package main

import (
	"context"
	"fmt"
	"os"

	log "github.com/Astera-org/easylog"
	"github.com/Astera-org/obelisk/infra/gengo/infra"
	"github.com/apache/thrift/lib/go/thrift"
)

type WorkerApp struct {
	binCache BinCache
	rootDir  string
	jobCzar  *infra.JobCzarClient
	context  context.Context
}

func (app *WorkerApp) Init() {
	jobCzarAddr := fmt.Sprint(gConfig.JOBCZAR_IP, ":", gConfig.JOBCZAR_PORT)
	log.Info("Connecting to JobCzar: ", jobCzarAddr)
	app.binCache.Init()
	app.context = context.Background()
	app.jobCzar = MakeClient(jobCzarAddr)
	app.rootDir, _ = os.Getwd()
	log.Info("Root dir: ", app.rootDir)

	os.Mkdir(gConfig.BINDIR, 0755)
	os.Mkdir(gConfig.JOBDIR, 0755)
}

func MakeClient(addr string) *infra.JobCzarClient {
	transportFactory := thrift.NewTBufferedTransportFactory(8192)
	transportSocket := thrift.NewTSocketConf(addr, nil)
	transport, _ := transportFactory.GetTransport(transportSocket)

	protocolFactory := thrift.NewTBinaryProtocolFactoryConf(nil)

	iprot := protocolFactory.GetProtocol(transport)
	oprot := protocolFactory.GetProtocol(transport)

	transport.Open()

	return infra.NewJobCzarClient(thrift.NewTStandardClient(iprot, oprot))
}

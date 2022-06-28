package main

import (
	"context"
	"fmt"
	"os"

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
	app.binCache.Init()
	app.context = context.Background()
	app.jobCzar = MakeClient(fmt.Sprint(gConfig.JOBCZAR_IP, ":", gConfig.JOBCZAR_PORT))
	app.rootDir, _ = os.Getwd()

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

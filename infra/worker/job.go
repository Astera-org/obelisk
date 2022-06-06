package main

import (
	"context"
	"fmt"
	"os"

	"github.com/Astera-org/obelisk/infra/gengo/infra"
	"github.com/apache/thrift/lib/go/thrift"
)

type Job struct {
	jobID     int
	agentName string
	worldName string
}

func (job *Job) fetchWork() error {
	var defaultCtx = context.Background()
	var jobCzar = MakeClient(gConfig.JOBCZAR_IP)
	infraJob, err := jobCzar.FetchWork(defaultCtx, gConfig.WORKER_NAME, gConfig.INSTANCE_NAME)

	if err != nil {
		return err
	}

	job.jobID = int(infraJob.JobID)
	job.agentName = infraJob.AgentName
	job.worldName = infraJob.WorldName
	return nil
}

// This is the dir that the process will run out of and that we will save all the Job specific files to
func (job *Job) createJobDir() {
	dirName := fmt.Sprint(gConfig.JOBDIR_ROOT, job.jobID)

	err := os.Mkdir(dirName, 0755)
	if err != nil {
		fmt.Println("Couldn't create dir", err)
	}

	err = os.Chdir(dirName)
	if err != nil {
		fmt.Println("Couldn't cd", err)
	}

}

func (job *Job) setCfgs() {
	// TODO
}

func (job *Job) readResults() {
	// TODO
}

func (job *Job) returnResults() error {
	// TODO
	return nil
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

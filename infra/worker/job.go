package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"time"

	"github.com/Astera-org/obelisk/infra/gengo/infra"
	"github.com/apache/thrift/lib/go/thrift"
)

type Job struct {
	jobID     int32
	agentName string
	worldName string
	agentDesc AgentDesc
	worldDesc WorldDesc
	agentCfg  string
	worldCfg  string
	result    infra.ResultWork
}

const (
	goodJob      int32 = 0
	jobFailed          = 1
	malformedJob       = 2
)

func (job *Job) fetchWork() error {
	var defaultCtx = context.Background()
	var jobCzar = MakeClient(gConfig.JOBCZAR_IP)
	infraJob, err := jobCzar.FetchWork(defaultCtx, gConfig.WORKER_NAME, gConfig.INSTANCE_NAME)
	if err != nil {
		return err
	}

	job.jobID = infraJob.JobID
	job.agentName = infraJob.AgentName
	job.worldName = infraJob.WorldName
	job.agentCfg = infraJob.AgentCfg
	job.worldCfg = infraJob.WorldCfg
	job.result.JobID = job.jobID

	var exists bool
	job.agentDesc, exists = gConfig.AGENTS[job.agentName]
	if !exists {
		job.result.Status = jobFailed
		return errors.New("Unknown Agent" + job.agentName)
	}

	job.worldDesc, exists = gConfig.WORLDS[job.worldName]
	if !exists {
		job.result.Status = jobFailed
		return errors.New("Unknown World" + job.worldName)
	}

	return nil
}

// This is the dir that the process will run out of and that we will save all the Job specific files to
func (job *Job) createJobDir() error {
	dirName := fmt.Sprint(gConfig.JOBDIR_ROOT, job.jobID)

	err := os.Mkdir(dirName, 0755)
	if err != nil {
		return err
	}

	err = os.Chdir(dirName)
	if err != nil {
		return err
	}
	return nil
}

func (job *Job) setCfgs() error {
	// write the cfgs to file
	agentFile, err := os.Create("agent.cfg")
	if err != nil {
		return err
	}
	defer agentFile.Close()
	agentFile.WriteString("GITHASH=\"" + job.agentDesc.GITHASH + "\"\n")
	dt := time.Now()
	agentFile.WriteString("JOBSTART=" + dt.Format(time.RFC822) + "\"\n")
	agentFile.WriteString("##### end manifest ####\n")
	agentFile.WriteString(job.agentCfg)
	agentFile.Close()

	worldFile, err := os.Create("world.cfg")
	if err != nil {
		return err
	}
	defer worldFile.Close()
	worldFile.WriteString("GITHASH=\"" + job.worldDesc.GITHASH + "\"\n")
	worldFile.WriteString(job.worldCfg)
	worldFile.Close()
	return nil
}

func (job *Job) returnResults() error {
	var defaultCtx = context.Background()
	var jobCzar = MakeClient(gConfig.JOBCZAR_IP)

	ok, err := jobCzar.SubmitResult_(defaultCtx, &job.result)

	if err != nil {
		return err
	}

	if ok {
		return nil
	}

	return errors.New("false")
}

// need to bail from one process if the other dies
func (job *Job) doJob() {

	agentCtx, agentCancel := context.WithCancel(context.Background())
	worldCtx, worldCancel := context.WithCancel(agentCtx)
	defer agentCancel()
	defer worldCancel()

	agentCmd := exec.CommandContext(agentCtx, job.agentDesc.PATH)
	worldCmd := exec.CommandContext(worldCtx, job.worldDesc.PATH)
	err := worldCmd.Start()
	if err != nil {
		fmt.Println("world:", err)
		job.result.Status = jobFailed
		return
	}
	err = agentCmd.Start()
	if err != nil {
		fmt.Println("agent:", err)
		job.result.Status = jobFailed
		return
	}

	go worldCmd.Wait()

	err = agentCmd.Wait()

	// cd back out of the job dir
	err2 := os.Chdir("..")
	if err2 != nil {
		fmt.Println("Couldn't cd ", err2)
	}

	if err != nil {
		fmt.Println("agent:", err)
		job.result.Status = jobFailed
		return
	}
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

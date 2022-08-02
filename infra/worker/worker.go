package main

import (
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/Astera-org/obelisk/infra/common"
	"github.com/Astera-org/obelisk/infra/gengo/infra"
	"github.com/apache/thrift/lib/go/thrift"

	log "github.com/Astera-org/easylog"
)

/*
1) fetches a job from the jobCzar
2) create the job dir
3) Creates the .cfg used for the spawns
4) Spawns the necessary processes
5) Read the spawn result report
6) Reports the result of the process to the jobCzar
7) delete the spawn result report

*/
var gConfig Config
var gApp WorkerApp
var VERSION string = "v0.1.0"

func main() {
	gConfig.Load()
	if (gConfig.INSTANCE_ID == 0) && len(os.Args) > 0 {
		v, _ := strconv.Atoi(os.Args[1])
		gConfig.INSTANCE_ID = int32(v)
	}

	err := log.Init(
		log.SetLevel(log.INFO),
		log.SetFileName("worker.log"),
	)
	if err != nil {
		panic(err)
	}

	gApp.Init()

	handler := RequestHandler{}
	thriftServer := MakeThriftServer(handler)

	log.Info("Listening on: ", gConfig.WORKER_BASE_PORT+gConfig.INSTANCE_ID)

	go thriftServer.Serve()

	go mainLoop()

	common.SignalHandler()

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
		case "a":
			gApp.job.abort()
		case "s":
			printJobDetails()
		case "v":
			fmt.Println("Version: ", VERSION)
		default:
			printHelp()
		}
	}
}

func printHelp() {
	fmt.Println("Valid Commands:")
	fmt.Println("q: quit")
	fmt.Println("s: print state of worker/job")
	fmt.Println("a: abort current job")
	fmt.Println("v: print version")
}

func printJobDetails() {
	if gApp.job == nil {
		fmt.Println("Unemployed")
		fmt.Println("Status: ", gApp.statusString)
	} else {
		fmt.Println("JobID: ", gApp.job.JobID)
		fmt.Println("AgentName: ", gApp.job.AgentName, ".", gApp.job.AgentVersion)
		fmt.Println("WorldName: ", gApp.job.WorldName, ".", gApp.job.WorldVersion)
		fmt.Println("Running for: TODO")
		fmt.Println("Score: ", gApp.job.Result.Score)
		fmt.Println("Steps: ", gApp.job.Result.Steps)
		fmt.Println("Seconds: ", gApp.job.Result.Seconds)

	}
}

func mainLoop() {
	var still bool = true

	for still {

		gApp.job = &Job{}
		fetchJob(gApp.job)

		err := gApp.job.createJobDirs()
		if err != nil {
			log.Error("Creating job dir err: ", err)
			still = false
		} else {
			err = gApp.job.setCfgs()
			if err != nil {
				log.Error("Setting cfgs err: ", err)
				still = false
			} else {
				gApp.job.doJob()
				readResults(gApp.job)
			}
		}

		returnResults(gApp.job)
		log.Info("job completed")
	}

	os.Exit(-1)
}

func fetchJob(job *Job) {
	var waitSeconds int = 1
	for true {
		log.Info("Fetching new job")
		gApp.statusString = "fetching job"

		err := job.fetchWork()
		log.Info("Job Fetched: ", job.JobID) // TEMP
		if err != nil {
			log.Error("Fetching err: ", err)
			gApp.statusString = "fetching err"
			wait(&waitSeconds)
		} else {
			return
		}
	}
}

func returnResults(job *Job) {
	gApp.statusString = "returning results"
	var waitSeconds int = 1
	for true {
		err := job.returnResults()
		if err != nil {
			log.Error("Results err: ", err)
			gApp.statusString = "returning results err"
			wait(&waitSeconds)
		} else {
			return
		}
	}
}

func wait(waitSeconds *int) {
	*waitSeconds *= 2
	if *waitSeconds > 60*10 {
		*waitSeconds = 60 * 10
	}
	time.Sleep(time.Duration(*waitSeconds) * time.Second)
}

func MakeThriftServer(handler infra.WorkerService) *thrift.TSimpleServer {
	transportFactory := thrift.NewTBufferedTransportFactory(8192)
	transport, _ := thrift.NewTServerSocket(fmt.Sprint(":", gConfig.WORKER_BASE_PORT+gConfig.INSTANCE_ID))
	processor := infra.NewWorkerServiceProcessor(handler)
	protocolFactory := thrift.NewTBinaryProtocolFactoryConf(nil)
	server := thrift.NewTSimpleServer4(processor, transport, transportFactory, protocolFactory)
	return server
}

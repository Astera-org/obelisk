package main

import (
	"fmt"
	"os"
	"time"

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

	err := log.Init(
		log.SetLevel(log.INFO),
		log.SetFileName("worker.log"),
	)
	if err != nil {
		panic(err)
	}

	gApp.Init()

	go mainLoop()

	for true {
		var command string
		fmt.Scan(&command)
		switch command {
		case "q":
			os.Exit(0)
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
	fmt.Println("v: print version")
}

func mainLoop() {
	var still bool = true

	for still {

		var job Job
		fetchJob(&job)

		err := job.createJobDirs()
		if err != nil {
			log.Error("Creating job dir err: ", err)
			still = false
		} else {
			err = job.setCfgs()
			if err != nil {
				log.Error("Setting cfgs err: ", err)
				still = false
			} else {
				job.doJob()
				readResults(&job)
			}
		}

		returnResults(&job)
		log.Info("job completed")
	}

	os.Exit(-1)
}

func fetchJob(job *Job) {
	var waitSeconds int = 1
	for true {
		log.Info("Fetching job: ", job.jobID)

		err := job.fetchWork()
		log.Info("Job Fetched: ", job.jobID) // TEMP
		if err != nil {
			log.Error("Fetching err: ", err)
			wait(&waitSeconds)
		} else {
			return
		}
	}
}

func returnResults(job *Job) {
	var waitSeconds int = 1
	for true {
		err := job.returnResults()
		if err != nil {
			log.Error("Results err: ", err)
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

package main

import (
	"fmt"
	"os"
	"time"
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

func main() {
	gConfig.Load()

	go mainLoop()

	for true {
		var command string
		fmt.Scan(&command)
		switch command {
		case "q":
			os.Exit(0)
		default:
			printHelp()
		}
	}
}

func printHelp() {
	fmt.Println("Valid Commands:")
	fmt.Println("q: quit")
}

func mainLoop() {
	var still bool = true

	for still {
		var job Job
		fetchJob(&job)

		err := job.createJobDir()
		if err != nil {
			fmt.Println("Creating job dir err: ", err)
			still = false
		} else {
			err = job.setCfgs()
			if err != nil {
				fmt.Println("Setting cfgs err: ", err)
				still = false
			} else {
				job.doJob()
				readResults(&job)
			}
		}

		returnResults(&job)
		fmt.Println("job completed")
	}

	os.Exit(-1)
}

func fetchJob(job *Job) {
	var waitSeconds int = 1
	for true {
		fmt.Println("Fetching job: ", job.jobID)

		err := job.fetchWork()
		fmt.Println("Job Fetched: ", job.jobID) // TEMP
		if err != nil {
			fmt.Println("Fetching err: ", err)
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
			fmt.Println("Results err: ", err)
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

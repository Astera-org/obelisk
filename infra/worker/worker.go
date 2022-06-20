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
	var waitSeconds int = 1

	for true {
		var job Job

		for true {
			fmt.Println("Fetching job")

			err := job.fetchWork()
			if err != nil {
				fmt.Println("Fetching err: ", err)
				increaseBackoff(&waitSeconds)
				time.Sleep(time.Duration(waitSeconds) * time.Second)
				continue
			} else {
				waitSeconds = 1
				break
			}

		}
		job.createJobDir()
		job.setCfgs()
		job.doJob()
		readResults(&job)

		for true {
			err := job.returnResults()
			if err != nil {
				fmt.Println("Results err: ", err)
				increaseBackoff(&waitSeconds)
				time.Sleep(time.Duration(waitSeconds) * time.Second)
				continue
			} else {
				waitSeconds = 1
				break
			}
		}

		fmt.Println("job completed")
	}
}

func increaseBackoff(waitSeconds *int) {
	*waitSeconds *= 2
	if *waitSeconds > 60*10 {
		*waitSeconds = 60 * 10
	}
}

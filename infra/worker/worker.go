package main

import (
	"fmt"
	"os/exec"
	"sync"
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

	var waitSeconds int = 1

	for true {
		fmt.Println("Fetching job")
		var job Job
		err := job.fetchWork()
		if err != nil {
			fmt.Println("Fetching err: ", err)
			increaseBackoff(&waitSeconds)
			time.Sleep(time.Duration(waitSeconds) * time.Second)
			continue
		}
		job.setCfgs()

		var waitGroup sync.WaitGroup
		waitGroup.Add(2)
		go spawnWorld(&waitGroup, &job)
		go spawnAgent(&waitGroup, &job)

		waitGroup.Wait()

		job.readResults()

		err = job.returnResults()
		if err != nil {
			fmt.Println("Results err: ", err)
			increaseBackoff(&waitSeconds)
			time.Sleep(time.Duration(waitSeconds) * time.Second)
			continue
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

func spawnAgent(waitGroup *sync.WaitGroup, job *Job) {
	agentDesc, exists := gConfig.AGENTS[job.agentName]
	if exists {
		_, err := exec.Command(agentDesc.PATH).Output()
		if err != nil {
			switch e := err.(type) {
			case *exec.Error:
				fmt.Println("failed executing:", err)
			case *exec.ExitError:
				fmt.Println("command exit rc =", e.ExitCode())
			default:
				panic(err)
			}
		}
	} else {
		fmt.Println("Unknown Agent: ", job.agentName)
	}

	waitGroup.Done()
}

func spawnWorld(waitGroup *sync.WaitGroup, job *Job) {
	worldDesc, exists := gConfig.WORLDS[job.worldName]
	if exists {
		_, err := exec.Command(worldDesc.PATH).Output()
		if err != nil {
			switch e := err.(type) {
			case *exec.Error:
				fmt.Println("failed executing:", err)
			case *exec.ExitError:
				fmt.Println("command exit rc =", e.ExitCode())
			default:
				panic(err)
			}
		}
	} else {
		fmt.Println("Unknown World: ", job.worldName)
	}

	waitGroup.Done()
}

package main

import (
	"fmt"
	"os/exec"
	"sync"
)

/*
1) fetches a job from the jobCzar
2) Creates the .cfg used for the spawns
3) Spawns the necessary processes
4) Read the spawn result report
5) Reports the result of the process to the jobCzar
6) delete the spawn result report
*/
var gConfig Config

func main() {
	gConfig.Load()

	for true {
		fmt.Println("Fetching job")
		var job Job
		job.fetchWork()
		job.setCfgs()

		var waitGroup sync.WaitGroup
		waitGroup.Add(2)
		go spawnWorld(&waitGroup, &job)
		go spawnAgent(&waitGroup, &job)

		waitGroup.Wait()

		job.readResults()

		job.returnResults()

		fmt.Println("job completed")
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
	worldDesc, exists := gConfig.WORLDS[job.envName]
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
		fmt.Println("Unknown World: ", job.envName)
	}

	waitGroup.Done()
}

package main

import (
	"encoding/json"
	"io/ioutil"
	"os"

	log "github.com/Astera-org/easylog"
)

// if result file isn't there tell server it failed
func readResults(job *Job) {
	if job.Result.Status == goodJob {

		os.Chdir(job.AgentWorkingDir)
		// read the result file
		file, err := os.Open("result.json")
		if err != nil {
			log.Error("Couldn't open result.json", err)
			job.Result.Status = jobFailed
			return
		}
		defer file.Close()
		// parse the result file
		byteValue, _ := ioutil.ReadAll(file)

		json.Unmarshal(byteValue, &job.Result)
		job.Result.Cycles = int32(float32(job.Result.Seconds) * gConfig.CPU_FACTOR)
	}
}

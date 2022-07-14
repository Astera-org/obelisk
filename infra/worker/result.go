package main

import (
	"encoding/json"
	"io/ioutil"
	"os"

	log "github.com/Astera-org/easylog"
)

// if result file isn't there tell server it failed
func readResults(job *Job) {
	if job.result.Status == goodJob {

		os.Chdir(job.agentWorkingDir)
		// read the result file
		file, err := os.Open("result.json")
		if err != nil {
			log.Error("Couldn't open result.json", err)
			job.result.Status = jobFailed
			return
		}
		defer file.Close()
		// parse the result file
		byteValue, _ := ioutil.ReadAll(file)

		json.Unmarshal(byteValue, &job.result)
		job.result.Cycles = int32(float32(job.result.Seconds) * gConfig.CPU_FACTOR)
	}
}

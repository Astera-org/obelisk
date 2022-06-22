package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/Astera-org/obelisk/infra"
)

// if result file isn't there tell server it failed
func readResults(job *Job) {
	if job.result.Status == goodJob {

		// read the result file
		file, err := os.Open("result.json")
		if err != nil {
			fmt.Println("Couldn't open result.json", err)
			job.result.Status = jobFailed
			return
		}
		defer file.Close()
		// parse the result file
		byteValue, _ := ioutil.ReadAll(file)

		var result infra.Result
		json.Unmarshal(byteValue, &result)
		job.result.Cycles = result.Cycles
		job.result.Score = result.Score
		job.result.Seconds = result.Seconds

	}
}

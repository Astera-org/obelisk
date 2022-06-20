package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
)

type Result struct {
	Seconds int32   `json:"seconds"`
	Cycles  int32   `json:"cycles"`
	Score   float64 `json:"score"`
}

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

		var result Result
		json.Unmarshal(byteValue, &result)
		job.result.Cycles = result.Cycles
		job.result.Score = result.Score
		job.result.Seconds = result.Seconds

	}
}

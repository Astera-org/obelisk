package infra

import (
	"encoding/json"
	"io/ioutil"
)

type Result struct {
	Seconds int32   `json:"seconds"`
	Cycles  int32   `json:"cycles"`
	Score   float64 `json:"score"`
}

func WriteResults(score float64, cycles int32, seconds int32) {

	result := Result{
		Seconds: seconds,
		Cycles:  cycles,
		Score:   score,
	}

	file, _ := json.MarshalIndent(result, "", " ")

	_ = ioutil.WriteFile("result.json", file, 0644)

}

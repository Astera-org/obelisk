package main

import (
	"math/rand"
	"time"

	log "github.com/Astera-org/easylog"
	"github.com/Astera-org/obelisk/infra"
)

func main() {
	err := log.Init(
		log.SetLevel(log.INFO),
		log.SetFileName("mock.log"),
	)
	if err != nil {
		panic(err)
	}
	log.Info("Hello World")
	time.Sleep(10 * time.Second)
	// set value to rand float
	rand.Seed(time.Now().UnixNano())
	score := rand.Float64()
	cycles := int32(rand.Intn(1000))
	seconds := int32(rand.Intn(1000))

	infra.WriteResults(score, cycles, seconds)
}

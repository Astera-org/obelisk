package main

import (
	"fmt"

	"github.com/BurntSushi/toml"
)

// LATER move parts of this to a separate package

type Config struct {
	GUI           bool
	PROFILE       bool
	WORKER        bool
	LIFETIME      int32 // how many times the world will call step before we exit
	INTERNAL_PORT int32
	TESTFILE      string // location of sample data to test the neural network, and performance characteristics

	//network hyperparameters
	ACTAVG_INIT float64 //
	LRATE       float64 //scales the amount of weight change
	//
	//LearningRate
	//ActAvg
	//

}

func (config *Config) Load() {

	config.setDefaults()
	_, err := toml.DecodeFile("package/default.cfg", &config)
	if err != nil {
		fmt.Println(err)
		fmt.Println("default.cfg not found")
	}
	_, err = toml.DecodeFile("protobrain.cfg", &config)
	if err != nil {
		fmt.Println(err)
		fmt.Println("protobrain.cfg not found")
	}
}

func (config *Config) setDefaults() {
	config.GUI = true
	config.PROFILE = false
	config.WORKER = false
	config.LIFETIME = 100
	config.INTERNAL_PORT = 9090
	config.TESTFILE = "testdata.csv"

	config.LRATE = .1
	config.ACTAVG_INIT = 0.06
}

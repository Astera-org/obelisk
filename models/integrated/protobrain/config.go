package main

import (
	"fmt"
	"os"

	"github.com/BurntSushi/toml"
)

// LATER move parts of this to a separate package

type Config struct {
	GUI           bool
	PROFILE       bool
	WORKER        bool
	LIFETIME      int32 // how many times the world will call step before we exit
	INTERNAL_PORT int32
	HISTORYFILE   string // location to log files over the history of an agent
	TESTFILE      string // location of sample data to test the neural network, and performance characteristics
}

func (config *Config) Load() {

	config.setDefaults()
	_, err := toml.DecodeFile("brain.cfg", &config)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		fmt.Fprintln(os.Stderr, "Using defaults")
	}
}

func (config *Config) setDefaults() {
	config.GUI = true
	config.PROFILE = false
	config.WORKER = false
	config.LIFETIME = 100
	config.INTERNAL_PORT = 9090
	config.HISTORYFILE = "history.csv"
	config.TESTFILE = "testdata.csv"
}

package main

import (
	"fmt"
	"os"

	"github.com/BurntSushi/toml"
)

// LATER move parts of this to a separate package

type Config struct {
	GUI      bool
	PROFILE  bool
	WORKER   bool
	LIFETIME int32 // how many times the world will call step before we exit
}

func (config *Config) Load() {

	_, err := toml.DecodeFile("brain.cfg", &config)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		fmt.Fprintln(os.Stderr, "Using defaults")

		config.GUI = true
		config.PROFILE = false
		config.WORKER = false
		config.LIFETIME = 100
	}
}

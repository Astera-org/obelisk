package main

import (
	"fmt"
	"os"

	"github.com/BurntSushi/toml"
)

// LATER move parts of this to a separate package

type Config struct {
	GUI     bool
	PROFILE bool
}

func (config *Config) Load() {

	_, err := toml.DecodeFile("package/fworld.cfg", &config)
	_, err = toml.DecodeFile("fworld.cfg", &config)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		fmt.Fprintln(os.Stderr, "Using defaults")

		config.GUI = true
		config.PROFILE = false
	}
}

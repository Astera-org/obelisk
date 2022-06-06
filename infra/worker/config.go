package main

import (
	"fmt"
	"os"

	"github.com/BurntSushi/toml"
)

// LATER move parts of this to a separate package
type AgentDesc struct
{
	PATH string
}
type WorldDesc struct {
	PATH string
}
type Config struct {
	FETCHWORK_URL	string
	RESULTS_URL		string
	AGENTS 			map[string]AgentDesc
	WORLDS 			map[string]WorldDesc
}

func (config *Config) Load() {

	_, err := toml.DecodeFile("worker.cfg", &config)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		fmt.Fprintln(os.Stderr, "Using defaults")

		config.FETCHWORK_URL = "127.0.0.1"
		config.RESULTS_URL ="127.0.0.1"
	}
}
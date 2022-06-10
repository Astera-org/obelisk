package main

import (
	"fmt"
	"os"

	"github.com/BurntSushi/toml"
)

// LATER move parts of this to a separate package
type AgentDesc struct {
	PATH string
}
type WorldDesc struct {
	PATH string
}
type Config struct {
	WORKER_NAME   string
	INSTANCE_NAME string
	JOBDIR_ROOT   string
	JOBCZAR_IP    string
	AGENTS        map[string]AgentDesc
	WORLDS        map[string]WorldDesc
}

func (config *Config) Load() {

	config.setDefaults()
	_, err := toml.DecodeFile("worker.cfg", &config)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		fmt.Fprintln(os.Stderr, "Using defaults")
	}
	config.ensureRequired()
}

func (config *Config) setDefaults() {
	config.WORKER_NAME = ""
	config.INSTANCE_NAME = "?"
	config.JOBDIR_ROOT = "jobs"
	config.JOBCZAR_IP = "127.0.0.1:9009"
}

func (config *Config) ensureRequired() {
	if config.WORKER_NAME == "" {
		fmt.Println("WORKER_NAME must be set")
		panic(-1)
	}
}

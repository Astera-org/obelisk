package main

import (
	"fmt"
	"os"

	"github.com/BurntSushi/toml"
)

type Config struct {
	WORKER_NAME   string
	INSTANCE_NAME string
	JOBDIR_ROOT   string
	BINDIR_ROOT   string
	JOBCZAR_IP    string
	JOBCZAR_PORT  int32
	BINSERVER_URL string
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
	config.JOBDIR_ROOT = "job"
	config.BINDIR_ROOT = "bins"
	config.JOBCZAR_IP = "127.0.0.1"
	config.BINSERVER_URL = "127.0.0.1:8080"
	config.JOBCZAR_PORT = 9009
}

func (config *Config) ensureRequired() {
	if config.WORKER_NAME == "" {
		fmt.Println("WORKER_NAME must be set")
		panic(-1)
	}
}

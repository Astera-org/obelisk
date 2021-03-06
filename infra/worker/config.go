package main

import (
	"fmt"
	"log"
	"os"

	"github.com/BurntSushi/toml"
)

type Config struct {
	WORKER_NAME      string
	INSTANCE_ID      int32
	WORKER_BASE_PORT int32
	JOBDIR           string
	JOBDIRPREFIX     string
	BINDIR           string
	JOBCZAR_IP       string
	JOBCZAR_PORT     int32
	BINSERVER_URL    string
	WINDOWS          bool
	CPU_FACTOR       float32
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
	config.INSTANCE_ID = 0
	config.JOBDIR = "jobs"
	config.JOBDIRPREFIX = "job"
	config.BINDIR = "bins"
	config.WINDOWS = false

	config.JOBCZAR_IP = "127.0.0.1"
	config.BINSERVER_URL = "127.0.0.1:9003"
	config.JOBCZAR_PORT = 9001
	config.WORKER_BASE_PORT = 9100
	config.CPU_FACTOR = 1
}

func (config *Config) ensureRequired() {
	if config.WORKER_NAME == "" {
		log.Fatal("WORKER_NAME must be set")
	}
}

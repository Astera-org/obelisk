package main

import (
	"fmt"
	"os"

	"github.com/BurntSushi/toml"
)

// this is loaded from the .cfg file
type Project struct {
	name              string
	path              string
	buildOperation    string
	binType           int
	regressionBinName string
	targetScore       float64
	scoreTolerance    float64
	targetCycles      float64
	cycleTolerance    float64
}

type Config struct {
	DB_CONNECT    string
	USER_ID       int
	TEMP_ROOT     string // where we put the temp directories
	REPO_NAME     string
	PROJECTS      map[string]Project
	SERVER_PORT   int32 // where it is listening for events from github
	WINDOWS       bool  // if we are running on windows
	CALLBACK_URL  string
	BINSERVER_LOC string
}

func (config *Config) Load() {

	config.setDefaults()
	_, err := toml.DecodeFile("odpw.cfg", &config)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		fmt.Fprintln(os.Stderr, "Using defaults")
	}
	config.ensureRequired()
}

func (config *Config) setDefaults() {
	config.DB_CONNECT = ""
	config.TEMP_ROOT = "temp"
	config.REPO_NAME = ""
	config.PROJECTS = make(map[string]Project)
	config.SERVER_PORT = 8080
	config.WINDOWS = false
	config.USER_ID = 0
	config.CALLBACK_URL = "http://localhost:8080/result"
	config.BINSERVER_LOC = "binserver"
}

func (config *Config) ensureRequired() {

}

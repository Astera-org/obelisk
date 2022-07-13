package main

import (
	"fmt"
	"os"
	"strings"

	log "github.com/Astera-org/easylog"
	"github.com/BurntSushi/toml"
)

// this is loaded from the .cfg file
type Project struct {
	Name              string
	Path              string
	BuildOperation    string
	BinType           int
	RegressionBinName string
	TargetScore       float64
	ScoreTolerance    float64
	TargetCycles      float64
	CycleTolerance    float64
}

type Config struct {
	DB_CONNECT    string
	USER_ID       int
	TEMP_ROOT     string // where we put the temp directories
	REPO_PATH     string
	BRANCH_NAME   string
	PROJECTS      map[string]Project
	SERVER_PORT   int32 // where it is listening for events from github
	WINDOWS       bool  // if we are running on windows
	CALLBACK_URL  string
	BINSERVER_LOC string
	REPO_NAME     string // Derived from the REPO_PATH
}

func (config *Config) Load() {

	config.setDefaults()
	_, err := toml.DecodeFile("odpw.cfg", &config)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		fmt.Fprintln(os.Stderr, "Using defaults")
	}
	config.ensureRequired()

	// extract the repo name from the repo path
	config.REPO_NAME = config.REPO_PATH[strings.LastIndex(config.REPO_PATH, "/")+1:]
	config.REPO_NAME = config.REPO_NAME[:len(config.REPO_NAME)-4]
}

func (config *Config) setDefaults() {
	config.DB_CONNECT = ""
	config.TEMP_ROOT = "temp"
	config.REPO_PATH = ""
	config.PROJECTS = make(map[string]Project)
	config.SERVER_PORT = 8080
	config.WINDOWS = false
	config.USER_ID = 0
	config.CALLBACK_URL = "http://localhost:8080/result"
	config.BINSERVER_LOC = "binserver"
	config.BRANCH_NAME = "master"
}

func (config *Config) ensureRequired() {

	if config.DB_CONNECT == "" {
		log.Fatal("DB_CONNECT must be set")
	}
	if config.REPO_PATH == "" {
		log.Fatal("REPO_PATH must be set")
	}
	if config.BRANCH_NAME == "" {
		log.Fatal("BRANCH_NAME must be set")
	}

}

package main

import (
	"fmt"
	"os"

	"github.com/BurntSushi/toml"
)

type Config struct {
	BINARY_ROOT    string
	TEMP_DIR       string
	COMPLETED_ROOT string
	DB_CONNECT     string
	SERVER_PORT    int
	IS_LOCALHOST   bool
}

func (config *Config) Load() {

	config.setDefaults()
	_, err := toml.DecodeFile("binserver.cfg", &config)

	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		fmt.Fprintln(os.Stderr, "Using defaults")
	}
	config.ensureRequired()
}

func (config *Config) setDefaults() {
	config.BINARY_ROOT = "binaries"
	config.COMPLETED_ROOT = "completed"
	config.DB_CONNECT = ""
	config.TEMP_DIR = "temp"

	config.SERVER_PORT = 8080
	config.IS_LOCALHOST = false
}

func (config *Config) ensureRequired() {

}

package main

import (
	"fmt"
	"os"

	"github.com/BurntSushi/toml"
)

type Config struct {
	BINARY_ROOT string
	TEMP_DIR    string
	DB_CONNECT  string
	SERVER_PORT string
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
	config.DB_CONNECT = ""
	config.TEMP_DIR = "temp"
	config.SERVER_PORT = "8080"
}

func (config *Config) ensureRequired() {

}
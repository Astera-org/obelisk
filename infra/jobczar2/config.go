package main

import (
	"fmt"
	"os"

	"github.com/BurntSushi/toml"
)

// LATER move parts of this to a separate package

type Config struct {
	DB_CONNECT  string
	SERVER_ADDR string
}

func (config *Config) Load() {

	config.setDefaults()
	_, err := toml.DecodeFile("jobczar.cfg", &config)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		fmt.Fprintln(os.Stderr, "Using defaults")
	}
	config.ensureRequired()
}

func (config *Config) setDefaults() {
	config.DB_CONNECT = ""
	config.SERVER_ADDR = "localhost:9009"
}

func (config *Config) ensureRequired() {

}

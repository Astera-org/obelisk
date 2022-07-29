package main

import (
	"fmt"
	"os"

	"github.com/BurntSushi/toml"
)

// LATER move parts of this to a separate package

type Config struct {
	DB_CONNECT   string
	THRIFT_PORT  int
	HTTP_PORT    int
	IS_LOCALHOST bool
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
	config.THRIFT_PORT = 9009
	config.HTTP_PORT = 8000
	config.IS_LOCALHOST = false
}

func (config *Config) ensureRequired() {

}

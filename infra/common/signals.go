package common

import (
	"os"
	"os/signal"
	"syscall"

	log "github.com/Astera-org/easylog"
)

func SignalHandler() {
	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)
	// this happens when running in background and the stdin closes
	// ERDAL: this doesn't work on Windows
	// TODO: use go build directives to split platform specific code
	// signal.Ignore(syscall.SIGURG, syscall.SIGTTIN)

	go func() {
		for sig := range sigs {
			log.Info("signalHandler received signal: ", sig)
			// the reason we need to custom handle this is goji intercepts it
			// but doesn't stop the entire process since we have multiple goroutines
			if sig == syscall.SIGINT || sig == syscall.SIGTERM {
				log.Info("exiting")
				os.Exit(0)
			}
		}
	}()
}

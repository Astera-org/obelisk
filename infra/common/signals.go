package common

import (
	log "github.com/Astera-org/easylog"
	"os"
	"os/signal"
	"syscall"
)

func SignalHandler() {
	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)
	// this happens when running in background and the stdin closes
	signal.Ignore(syscall.SIGURG, syscall.SIGTTIN)

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

package common

import (
	"golang.org/x/sync/errgroup"
	"net/http"

	log "github.com/Astera-org/easylog"
)

// StartHttpServer starts an http server and listens for errors
// a common failure mode is if the port is already taken
func StartHttpServer(serverAddr string, handler http.Handler) {
	log.Info("starting http server on: ", serverAddr)
	var g errgroup.Group
	g.Go(func() error {
		return http.ListenAndServe(serverAddr, handler)
	})
	err := g.Wait()
	if err != nil {
		log.Error("http server failure: ", err)
		panic(err)
	}
}

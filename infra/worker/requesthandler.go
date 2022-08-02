package main

import (
	"context"

	"github.com/Astera-org/obelisk/infra/gengo/infra"
)

type RequestHandler struct {
	infra.WorkerService
}

func (handler RequestHandler) StopJob(ctx context.Context, jobID int32) error {
	// make sure this is the job we are running
	// kill it
	// LATER
	//gApp.
	return nil
}

func (handler RequestHandler) SendResults(ctx context.Context, jobID int32) error {
	// find archive
	// send to binserver
	// LATER
	return nil
}

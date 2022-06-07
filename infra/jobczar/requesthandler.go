package main

import (
	"context"

	"github.com/Astera-org/obelisk/infra/gengo/infra"
)

type RequestHandler struct {
	infra.JobCzar
}

func (agent RequestHandler) FetchWork(ctx context.Context, workerName string, instanceName string) (*infra.Job, error) {
	// TODO
	job := infra.Job{}
	return &job, nil
}

func (agent RequestHandler) submitResult(ctx context.Context, result infra.ResultWork) (bool, error) {
	// TODO

	return true, nil
}

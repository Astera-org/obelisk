package main

import (
	"errors"
	"fmt"

	commonInfra "github.com/Astera-org/obelisk/infra"
)

type BinServerApp struct {
	currentlyFetching map[int]bool
	db                Database
}

func (app *BinServerApp) Init() {
	app.currentlyFetching = make(map[int]bool)
	app.db.Connect()
}

func (app *BinServerApp) fetchRunResult(jobID int) error {

	// make sure we aren't already fetching this job
	if _, ok := app.currentlyFetching[jobID]; ok {
		// we are already fetching this job
		// LATER wait on the other thread
	}

	// make sure this job is complete
	status, workerName, _ := app.db.getJobInfo(jobID)
	if status != 2 {

		return errors.New("Job not completed")
	}

	// call ansible to fetch the job
	// LATER: add this threadID or something so other people can wait on this
	// LATER: fix the cmd
	cmdStr := fmt.Sprint("ansible fetch ", workerName, " ", jobID)
	commonInfra.RunCommand(cmdStr)

	return nil
}

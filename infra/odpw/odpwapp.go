package main

import (
	"fmt"
	"os"
	"sync"

	log "github.com/Astera-org/easylog"
	"github.com/Astera-org/obelisk/infra/gengo/infra"
)

type OdpwApp struct {
	rootDir  string
	index    int
	mutex    sync.Mutex
	database Database
	jobs     map[int32]*Binary
}

func (app *OdpwApp) Init() {
	app.rootDir, _ = os.Getwd()
	app.jobs = make(map[int32]*Binary)

	err := os.MkdirAll(app.rootDir+"/"+gConfig.TEMP_ROOT, 0755)
	if err != nil {
		log.Error(err)
		panic(-1)
	}
	app.index = 0
	app.database.Connect()
}

func (app *OdpwApp) nextIndex() int {
	app.index++
	return app.index
}

func (app *OdpwApp) notifyAutorities(result *infra.ResultJob, message string) {
	log.Error(message)
	// TODO: email us
	//subject := "Message from ODPW"
	body := message + "\r\n"
	if result != nil {
		body += fmt.Sprintln("JobID: ", result.JobID)
		body += fmt.Sprintln("Status: ", result.Status)
		body += fmt.Sprintln("Seconds: ", result.Seconds)
		body += fmt.Sprintln("Steps: ", result.Steps)
		body += fmt.Sprintln("Cycles: ", result.Cycles)
		body += fmt.Sprintln("Score: ", result.Score)
		body += fmt.Sprintln("WorkerName: ", result.WorkerName)
		body += fmt.Sprintln("InstanceID: ", result.InstanceID)
	}
}

// Add Job to the database and the jobs map
func (app *OdpwApp) AddJob(bin *Binary, agentID int, worldID int) {
	jobID, err := app.database.AddJob(gConfig.USER_ID, agentID, worldID)
	if err != nil {
		log.Error("AddJob: ", err)
		return
	}

	app.jobs[jobID] = bin
}

func (app *OdpwApp) processJobResult(jobResult *infra.ResultJob) {
	if bin, ok := app.jobs[jobResult.JobID]; ok {
		bin.gotJobResult(jobResult)
		delete(app.jobs, jobResult.JobID)
	} else {
		log.Info("Job not found: ", jobResult.JobID)
	}
}

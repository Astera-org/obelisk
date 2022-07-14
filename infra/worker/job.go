package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"time"

	log "github.com/Astera-org/easylog"

	"github.com/Astera-org/obelisk/infra/gengo/infra"
)

type Job struct {
	jobID           int32
	agentName       string
	worldName       string
	agentVersion    string
	worldVersion    string
	agentCfg        string
	worldCfg        string
	agentBinPath    string // path to the actual binary
	worldBinPath    string
	agentWorkingDir string
	worldWorkingDir string
	result          infra.ResultJob
}

const (
	goodJob      int32 = 0
	jobFailed          = 1
	malformedJob       = 2
)

func (job *Job) fetchWork() error {
	infraJob, err := gApp.jobCzar.FetchWork(gApp.context, gConfig.WORKER_NAME, gConfig.INSTANCE_NAME)
	if err != nil {
		return err
	}

	// this will download the binary if we don't have it locally
	agentBinInfo := gApp.binCache.EnsureBinary(infraJob.AgentID)
	if agentBinInfo == nil {
		job.result.Status = jobFailed
		return errors.New(fmt.Sprint("agent not found", infraJob.AgentID))
	}
	job.agentBinPath = getLocalPath(agentBinInfo)
	worldBinInfo := gApp.binCache.EnsureBinary(infraJob.WorldID)

	job.jobID = infraJob.JobID
	job.agentName = agentBinInfo.Name
	job.agentVersion = agentBinInfo.Version
	if worldBinInfo != nil {
		job.worldName = worldBinInfo.Name
		job.worldVersion = worldBinInfo.Version
		job.worldBinPath = getLocalPath(worldBinInfo)
	}
	job.agentCfg = infraJob.AgentCfg
	job.worldCfg = infraJob.WorldCfg

	job.result.JobID = job.jobID

	return nil
}

func getLocalPath(binInfo *infra.BinInfo) string {
	if gConfig.WINDOWS {
		return gApp.rootDir + "/" + gConfig.BINDIR + "/" + binInfo.Name + "/" + binInfo.Version + "/binary.exe"
	}

	return gApp.rootDir + "/" + gConfig.BINDIR + "/" + binInfo.Name + "/" + binInfo.Version + "/binary"

}

// This is the dir that the process will run out of and that we will save all the Job specific files to
func (job *Job) createJobDirs() error {
	jobRootDir := fmt.Sprint(gApp.rootDir+"/"+gConfig.JOBDIR+"/"+gConfig.JOBDIRPREFIX, job.jobID)

	job.agentWorkingDir = jobRootDir + "/agent"
	job.worldWorkingDir = jobRootDir + "/world"

	err := os.MkdirAll(job.agentWorkingDir, 0755)
	if err != nil {
		job.result.Status = jobFailed
		log.Error("createJobDir1:", err)
		return err
	}
	os.Symlink(gApp.rootDir+"/"+gConfig.BINDIR+"/"+job.agentName+"/data", job.agentWorkingDir+"/data")
	os.Symlink(gApp.rootDir+"/"+gConfig.BINDIR+"/"+job.agentName+"/"+job.agentVersion, job.agentWorkingDir+"/package")

	err = os.MkdirAll(job.worldWorkingDir, 0755)
	if err != nil {
		job.result.Status = jobFailed
		log.Error("createJobDir1:", err)
		return err
	}
	os.Symlink(gApp.rootDir+"/"+gConfig.BINDIR+"/"+job.worldName+"/data", job.worldWorkingDir+"/data")
	os.Symlink(gApp.rootDir+"/"+gConfig.BINDIR+"/"+job.worldName+"/"+job.worldVersion, job.worldWorkingDir+"/package")

	return nil
}

func (job *Job) setCfgs() error {
	// write the cfgs to file
	os.Chdir(job.agentWorkingDir)
	agentFile, err := os.Create(job.agentName + ".cfg")
	if err != nil {
		job.result.Status = jobFailed
		return err
	}
	defer agentFile.Close()
	agentFile.WriteString("NAME=\"" + job.agentName + "\"\n")
	agentFile.WriteString("VERSION=\"" + job.agentVersion + "\"\n")
	dt := time.Now()
	agentFile.WriteString("JOBSTART=\"" + dt.Format(time.RFC1123) + "\"\n")
	agentFile.WriteString("WORKER=true\n")
	agentFile.WriteString("##### end manifest ####\n\n")
	agentFile.WriteString(job.agentCfg)

	if job.worldName != "" {
		os.Chdir(job.worldWorkingDir)
		worldFile, err := os.Create(job.worldName + ".cfg")
		if err != nil {
			job.result.Status = jobFailed
			return err
		}
		defer worldFile.Close()
		worldFile.WriteString("NAME=\"" + job.worldName + "\"\n")
		worldFile.WriteString("VERSION=\"" + job.worldVersion + "\"\n")
		worldFile.WriteString("WORKER=true\n")
		worldFile.WriteString("##### end manifest ####\n\n")
		worldFile.WriteString(job.worldCfg)
	}
	return nil
}

func (job *Job) returnResults() error {
	ok, err := gApp.jobCzar.SubmitResult_(gApp.context, &job.result)

	if err != nil {
		return err
	}

	if ok {
		return nil
	}

	return errors.New("false")
}

// need to bail from one process if the other dies
func (job *Job) doJob() {
	agentCtx, agentCancel := context.WithCancel(context.Background())
	defer agentCancel()
	os.Chdir(job.agentWorkingDir)
	agentCmd := exec.CommandContext(agentCtx, job.agentBinPath)

	if job.worldName != "" {
		worldCtx, worldCancel := context.WithCancel(agentCtx)
		defer worldCancel()
		os.Chdir(job.worldWorkingDir)
		worldCmd := exec.CommandContext(worldCtx, job.worldBinPath)
		err := worldCmd.Start()
		if err != nil {
			log.Error("world:", err)
			job.result.Status = jobFailed
			return
		}

		go worldCmd.Wait()
	}

	err := agentCmd.Start()
	if err != nil {
		log.Error("agent:", err)
		job.result.Status = jobFailed
		return
	}

	err = agentCmd.Wait()
	if err != nil {
		log.Error("agent:", err)
		job.result.Status = jobFailed
		return
	}
}

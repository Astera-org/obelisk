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
	infra.Job

	AgentName       string
	WorldName       string
	AgentVersion    string
	WorldVersion    string
	AgentBinPath    string // path to the actual binary
	WorldBinPath    string
	AgentWorkingDir string
	WorldWorkingDir string
	Result          infra.ResultJob
}

const (
	goodJob      int32 = 0
	jobFailed          = 1
	malformedJob       = 2
)

func (job *Job) fetchWork() error {
	infraJob, err := gApp.jobCzar.FetchWork(gApp.context, gConfig.WORKER_NAME, gConfig.INSTANCE_ID)
	if err != nil {
		return err
	}

	gApp.statusString = "job fetched"

	// this will download the binary if we don't have it locally
	agentBinInfo := gApp.binCache.EnsureBinary(infraJob.AgentID)
	if agentBinInfo == nil {
		job.Result.Status = jobFailed
		return errors.New(fmt.Sprint("agent not found ", infraJob.AgentID))
	}
	job.AgentBinPath = getLocalPath(agentBinInfo)
	worldBinInfo := gApp.binCache.EnsureBinary(infraJob.WorldID)

	job.JobID = infraJob.JobID
	job.AgentName = agentBinInfo.Name
	job.AgentVersion = agentBinInfo.Version
	if worldBinInfo != nil {
		job.WorldName = worldBinInfo.Name
		job.WorldVersion = worldBinInfo.Version
		job.WorldBinPath = getLocalPath(worldBinInfo)
	}
	job.AgentParam = infraJob.AgentParam
	job.WorldParam = infraJob.WorldParam

	job.Result.JobID = job.JobID

	return nil
}

func getLocalPath(binInfo *infra.BinInfo) string {
	if gConfig.WINDOWS {
		return gApp.rootDir + "/" + gConfig.BINDIR + "/" + binInfo.Name + "/" + binInfo.Version + "/" + binInfo.Name + ".exe"
	}

	return gApp.rootDir + "/" + gConfig.BINDIR + "/" + binInfo.Name + "/" + binInfo.Version + "/" + binInfo.Name

}

// This is the dir that the process will run out of and that we will save all the Job specific files to
func (job *Job) createJobDirs() error {
	jobRootDir := fmt.Sprint(gApp.rootDir+"/"+gConfig.JOBDIR+"/"+gConfig.JOBDIRPREFIX, job.JobID)

	job.AgentWorkingDir = jobRootDir + "/agent"
	job.WorldWorkingDir = jobRootDir + "/world"

	err := os.MkdirAll(job.AgentWorkingDir, 0755)
	if err != nil {
		job.Result.Status = jobFailed
		log.Error("createJobDir1:", err)
		return err
	}
	os.Symlink(gApp.rootDir+"/"+gConfig.BINDIR+"/"+job.AgentName+"/data", job.AgentWorkingDir+"/data")
	os.Symlink(gApp.rootDir+"/"+gConfig.BINDIR+"/"+job.AgentName+"/"+job.AgentVersion, job.AgentWorkingDir+"/package")

	err = os.MkdirAll(job.WorldWorkingDir, 0755)
	if err != nil {
		job.Result.Status = jobFailed
		log.Error("createJobDir1:", err)
		return err
	}
	os.Symlink(gApp.rootDir+"/"+gConfig.BINDIR+"/"+job.WorldName+"/data", job.WorldWorkingDir+"/data")
	os.Symlink(gApp.rootDir+"/"+gConfig.BINDIR+"/"+job.WorldName+"/"+job.WorldVersion, job.WorldWorkingDir+"/package")

	return nil
}

func (job *Job) setCfgs() error {
	// write the cfgs to file
	os.Chdir(job.AgentWorkingDir)
	agentFile, err := os.Create(job.AgentName + ".cfg")
	if err != nil {
		job.Result.Status = jobFailed
		return err
	}

	defer agentFile.Close()
	agentFile.WriteString(fmt.Sprint("JOBID=\"", job.JobID, "\"\n"))
	agentFile.WriteString("NAME=\"" + job.AgentName + "\"\n")
	agentFile.WriteString("VERSION=\"" + job.AgentVersion + "\"\n")
	dt := time.Now()
	agentFile.WriteString("JOBSTART=\"" + dt.Format(time.RFC1123) + "\"\n")
	agentFile.WriteString("WORKER=true\n")
	agentFile.WriteString("##### end manifest ####\n\n")
	agentFile.WriteString(job.AgentParam)

	if job.WorldName != "" {
		os.Chdir(job.WorldWorkingDir)
		worldFile, err := os.Create(job.WorldName + ".cfg")
		if err != nil {
			job.Result.Status = jobFailed
			return err
		}
		defer worldFile.Close()
		worldFile.WriteString("NAME=\"" + job.WorldName + "\"\n")
		worldFile.WriteString("VERSION=\"" + job.WorldVersion + "\"\n")
		worldFile.WriteString("WORKER=true\n")
		worldFile.WriteString("##### end manifest ####\n\n")
		worldFile.WriteString(job.WorldParam)
	}
	return nil
}

func (job *Job) returnResults() error {
	ok, err := gApp.jobCzar.SubmitResult_(gApp.context, &job.Result)

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
	gApp.statusString = "job started"

	agentCtx, agentCancel := context.WithCancel(context.Background())
	defer agentCancel()
	os.Chdir(job.AgentWorkingDir)
	agentCmd := exec.CommandContext(agentCtx, job.AgentBinPath)

	if job.WorldName != "" {
		worldCtx, worldCancel := context.WithCancel(agentCtx)
		defer worldCancel()
		os.Chdir(job.WorldWorkingDir)
		worldCmd := exec.CommandContext(worldCtx, job.WorldBinPath)
		err := worldCmd.Start()
		if err != nil {
			log.Error("world:", err)
			job.Result.Status = jobFailed
			return
		}

		go worldCmd.Wait()
	}

	err := agentCmd.Start()
	if err != nil {
		log.Error("agent:", err)
		job.Result.Status = jobFailed
		return
	}

	err = agentCmd.Wait()
	if err != nil {
		log.Error("agent:", err)
		job.Result.Status = jobFailed
		return
	}
}

func (job *Job) abort() {
	// TODO
	gApp.statusString = "aborting job"
	log.Info("Aborting current job")

}

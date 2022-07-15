package main

import (
	"fmt"
	"os"

	log "github.com/Astera-org/easylog"
	"github.com/Astera-org/obelisk/infra"
	"github.com/Astera-org/obelisk/infra/gengo/infra"
)

type Binary struct {
	project     *Project
	binaryID    int // the ID in the database
	packageHash string
	version     string
}

// Init the ProjectTest
func (bin *Binary) Init(project *Project, version string) {
	bin.binaryID = 0
	bin.project = project
	bin.version = version
}

// build the binary and hash the package
func (bin *Binary) build(tempDir string) error {
	gApp.mutex.Lock()
	defer gApp.mutex.Unlock()

	packageRoot := tempDir + "/" + gConfig.REPO_NAME + "/" + bin.project.Path
	log.Info("Building binary: ", packageRoot)
	err := os.Chdir(packageRoot)
	if err != nil {
		log.Error("Could not change directory: ", err)
		return err
	}

	err = commonInfra.RunCommand(bin.project.BuildOperation)
	if err != nil {
		log.Error("Error building binary: ", err)
		return err
	}
	binaryName := bin.project.Name

	if gConfig.WINDOWS {
		binaryName += ".exe"
	}

	bin.packageHash, err = commonInfra.CalculatePackageHash(binaryName, packageRoot)
	if err != nil {
		log.Error("Error hashing binary: ", err)
		return err
	}

	return nil
}

// upload binary and the package dir to the binserver
func (bin *Binary) uploadFiles(tempDir string) error {
	localDir := tempDir + "/" + gConfig.REPO_NAME + "/" + bin.project.Path

	remotePath := gConfig.BINSERVER_LOC + "/" + bin.project.Name + "/" + bin.version
	// LATER: don't asssume binserver is local
	err := commonInfra.RunCommand("mkdir -p " + remotePath)
	if err != nil {
		return err
	}

	// copy the binary
	scp := fmt.Sprintf("scp %s/%s %s/%s",
		localDir, bin.project.Name, remotePath, bin.project.Name)
	log.Info("Uploading binary: ", scp)
	err = commonInfra.RunCommand(scp)
	if err != nil {
		return err
	}

	// check to see if the package dir exists
	_, err = os.Stat(localDir + "/package")
	if err == nil {
		// copy all of the package dir
		scp = fmt.Sprintf("scp -r %s/package %s/%s/%s",
			localDir, gConfig.BINSERVER_LOC, bin.project.Name, bin.version)
		log.Info("Uploading binary: ", scp)
		err = commonInfra.RunCommand(scp)
		return err
	}
	return nil
}

func (bin *Binary) addToDB() error {
	// create new entry in the DB
	var err error
	bin.binaryID, err = gApp.database.AddBinary(bin.project.Name, bin.version, bin.packageHash, bin.project.BinType)
	if err != nil {
		return err
	}
	return nil
}

func (bin *Binary) addTestJob() error {
	otherID, err := gApp.database.GetLatestID(bin.project.RegressionBinName)
	if err != nil {
		return err
	}
	var agentID int
	var worldID int
	if bin.project.BinType == 0 {
		agentID = bin.binaryID
		worldID = otherID
	} else {
		agentID = otherID
		worldID = bin.binaryID
	}
	gApp.AddJob(bin, agentID, worldID)
	return nil
}

func (bin *Binary) gotJobResult(result *infra.ResultJob) {
	var newStatus int = 0
	if bin.project.TargetScore*bin.project.ScoreTolerance > result.Score {
		errorStr := fmt.Sprint("Score too low: ", result.Score)
		gApp.notifyAutorities(result, errorStr)
		newStatus = 3
	}

	if int32(bin.project.TargetCycles*bin.project.CycleTolerance) < result.Cycles {
		errorStr := fmt.Sprint("Compute too high: ", result.Cycles)
		gApp.notifyAutorities(result, errorStr)
		newStatus = 3
	}

	gApp.database.SetStatus(bin.binaryID, newStatus)

}

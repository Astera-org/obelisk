package main

import (
	"fmt"
	"os"
	"os/exec"

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

	packageRoot := tempDir + "/" + bin.project.path

	os.Chdir(packageRoot)
	cmd := exec.Command(bin.project.buildOperation)
	err := cmd.Run()
	if err != nil {
		return err
	}
	binaryName := bin.project.name

	if gConfig.WINDOWS {
		binaryName += ".exe"
	}

	// add all the files in the package dir to fileList
	list, err := os.ReadDir(packageRoot + "/package")

	var fileList []string = make([]string, len(list)+1)
	fileList[0] = binaryName
	for n, dirItem := range list {
		fileList[n+1] = "/package/" + dirItem.Name()
	}

	bin.packageHash, err = commonInfra.HashFileList(packageRoot, fileList)
	if err != nil {
		return err
	}

	return nil
}

// upload binary and the package dir to the binserver
func (bin *Binary) uploadFiles(tempDir string) {
	localDir := tempDir + "/" + bin.project.path

	// copy the binary
	scp := fmt.Sprintf("scp %s/%s %s/%s/%s/%s",
		localDir, bin.project.name, gConfig.BINSERVER_LOC, bin.project.name, bin.version, bin.project.name)
	exec.Command(scp).Run()

	// check to see if the package dir exists
	_, err := os.Stat(localDir + "/package")
	if err == nil {
		// copy all of the package dir
		scp = fmt.Sprintf("scp -r %s/package %s/%s/%s",
			localDir, gConfig.BINSERVER_LOC, bin.project.name, bin.version)

		exec.Command(scp).Run()
	}
}

func (bin *Binary) addToDB() error {
	// create new entry in the DB
	var err error
	bin.binaryID, err = gApp.database.AddBinary(bin.project.name, bin.version, bin.packageHash, bin.project.binType)
	if err != nil {
		return err
	}
	return nil
}

func (bin *Binary) addTestJob() error {
	otherID, err := gApp.database.GetLatestID(bin.project.regressionBinName)
	if err != nil {
		return err
	}
	var agentID int
	var worldID int
	if bin.project.binType == 0 {
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
	if bin.project.targetScore*bin.project.scoreTolerance > result.Score {
		errorStr := fmt.Sprint("Score too low: ", result.Score)
		gApp.notifyAutorities(result, errorStr)
		newStatus = 3
	}

	if int32(bin.project.targetCycles*bin.project.cycleTolerance) < result.Cycles {
		errorStr := fmt.Sprint("Compute too high: ", result.Cycles)
		gApp.notifyAutorities(result, errorStr)
		newStatus = 3
	}

	gApp.database.SetStatus(bin.binaryID, newStatus)

}

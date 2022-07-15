package main

import (
	"fmt"
	"os"
	"os/exec"
	"strings"

	log "github.com/Astera-org/easylog"
)

// do everything we need to do for a given push to master
/*
- when something is pushed to master:
	- create temp dir
	- Checkout the repo
	- compile all projects we care about
	- email and stop on compile failures
	- compare the hashes to the hashes in the DB
	- if new then make new DB entries
	- move binaries to binserver
	- clean up temp dir
	- Kick off some regression tests
	- mark them as failed if so
	- send emails to the people


	How do we know when regression tests are done?
	How do we get the other files to run the projects?
		- the ones that are common for all of a given name should already be on the binserver
*/

type Push struct {
	tempDir   string
	shortHash string
}

func (push *Push) startPush(pushName string) {
	log.Info("Starting push: ", pushName)

	push.tempDir = gApp.rootDir + "/" + gConfig.TEMP_ROOT + "/" + pushName
	// if this dir already exists there is a problem
	_, err := os.Stat(push.tempDir)
	if err == nil {
		gApp.notifyAutorities(nil, push.tempDir+" alreday exists?")
		return
	}

	os.MkdirAll(push.tempDir, 0755)
	err = push.checkoutRepo(push.tempDir)
	if err != nil {
		log.Error("checkoutRepo: ", err)
		return
	}

	if !(pushName == push.shortHash) {
		gApp.notifyAutorities(nil, "Repo hashes don't match: "+pushName+" != "+push.shortHash)
		return
	}

	for _, project := range gConfig.PROJECTS {
		log.Info("Building project: ", project.Name)
		bin := &Binary{}
		bin.Init(&project, push.shortHash)

		err := bin.build(push.tempDir)
		if err != nil {
			errorStr := fmt.Sprint("buildProject: ", project, " ", err)
			gApp.notifyAutorities(nil, errorStr)
			continue
		}

		// see if exists in the DB
		if gApp.database.DoesExist(bin.packageHash) {
			log.Info("hash exists for: ", bin.project.Name, " ", bin.packageHash)
			continue
		} else {
			log.Info("Uploading new binary: ", bin.project.Name, " ", bin.version)
			err = bin.uploadFiles(push.tempDir)
			if err != nil {
				log.Error("uploadFiles: ", err)
				continue
			}
			err = bin.addToDB()
			if err != nil {
				log.Error("addToDB: ", err)
				continue
			}
			err = bin.addTestJob()
			if err != nil {
				log.Error("addTestJob: ", err)
				continue
			}
		}
	}
}

func (push *Push) checkoutRepo(tempDir string) error {
	log.Info("Checking out repo: ", tempDir)
	gApp.mutex.Lock()
	defer gApp.mutex.Unlock()
	err := os.Chdir(tempDir)
	if err != nil {
		log.Error("checkout: ", err)
		return err
	}

	err = exec.Command("git", "clone", gConfig.REPO_PATH).Run()
	if err != nil {
		log.Error("coRepo: ", err)
		return err
	}

	os.Chdir(gConfig.REPO_NAME)
	if gConfig.BRANCH_NAME != "master" {
		exec.Command("git", "checkout", gConfig.BRANCH_NAME).Run()
	}

	// run the setup.sh file if it exists
	_, err = os.Stat("setup.sh")
	if err == nil {
		// Repos should add this script if there is setup to be done besides just pulling the repo
		err = exec.Command("./setup.sh").Run()
		if err != nil {
			log.Error("setup: ", err)
			return err
		}
	}

	out, err := exec.Command("git", "rev-parse", "--short", "HEAD").Output()
	if err != nil {
		log.Error("coRepo: ", out)
		return err
	}
	push.shortHash = strings.TrimSpace(string(out))
	return nil
}

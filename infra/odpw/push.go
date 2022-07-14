package main

import (
	"fmt"
	"os"
	"os/exec"

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
	push.tempDir = gApp.rootDir + "/" + gConfig.TEMP_ROOT + "/" + pushName
	// if this dir already exists there is a problem
	_, err := os.Stat(push.tempDir)
	if err == nil {
		gApp.notifyAutorities(nil, push.tempDir+" alreday exists?")
		return
	}

	os.MkdirAll(push.tempDir, 0755)

	gApp.mutex.Lock()
	os.Chdir(push.tempDir)
	err = push.checkoutRepo(gConfig.REPO_NAME)
	gApp.mutex.Unlock()

	if pushName != push.shortHash {
		gApp.notifyAutorities(nil, "Repo hashes don't match: "+pushName+" != "+push.shortHash)
		return
	}

	if err != nil {
		log.Error("checkoutRepo: ", err)
		return
	}

	for _, project := range gConfig.PROJECTS {
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
			log.Info("hash exists for: ", bin.project.name)
			continue
		} else {
			bin.uploadFiles(push.tempDir)
			bin.addToDB()
			bin.addTestJob()
		}
	}
}

func (push *Push) checkoutRepo(repo string) error {
	exec.Command("git clone ", repo)

	out, err := exec.Command("git rev-parse --short HEAD").Output()
	if err != nil {
		return err
	}
	push.shortHash = string(out)
	return nil
}

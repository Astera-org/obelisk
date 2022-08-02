package main

import (
	"encoding/json"
	"fmt"
	"github.com/Astera-org/obelisk/infra/common"
	"net/http"
	"os"
	"time"

	log "github.com/Astera-org/easylog"
	"github.com/Astera-org/obelisk/infra/gengo/infra"
)

var gConfig Config
var gApp OdpwApp
var VERSION string = "v0.1.0"

func main() {
	err := log.Init(
		log.SetLevel(log.INFO),
		log.SetFileName("odpw.log"),
	)
	if err != nil {
		panic(err)
	}

	gConfig.Load()

	log.Info("Version: ", VERSION)
	log.Info("Repo: ", gConfig.REPO_PATH)
	log.Info("Name: ", gConfig.REPO_NAME)
	log.Info("Branch: ", gConfig.BRANCH_NAME)

	gApp.Init()

	http.HandleFunc("/result", jobResult)
	http.HandleFunc("/push", gitPush)
	var serverAddr = fmt.Sprint(":", gConfig.SERVER_PORT)
	go common.StartHttpServer(serverAddr, nil)

	common.SignalHandler()

	for true {
		var command string
		fmt.Scan(&command)
		if len(command) == 0 {
			// this happens when we try to run the process in the background
			time.Sleep(10 * time.Second)
			continue
		}
		switch command {
		case "q":
			os.Exit(0)
		case "t":
			test()
		case "f":
			forceStart()
		case "v":
			fmt.Println("Version: ", VERSION)
		default:
			printHelp()
		}
	}
}

func printHelp() {
	fmt.Println("Valid Commands:")
	fmt.Println("q: quit")
	fmt.Println("t: test")
	fmt.Println("f: force start")
	fmt.Println("v: print version")
}

func test() {
	log.Info("Test")

}

type GitCommitPaylod struct {
	Sha string `json:"sha"`
}
type GitBranchPaylod struct {
	Commit GitCommitPaylod `json:"commit"`
}

func forceStart() {
	// get from url https://api.github.com/repos/Astera-org/obelisk/branches/master
	url := "https://api.github.com/repos/Astera-org/obelisk/branches/" + gConfig.BRANCH_NAME
	resp, err := http.Get(url)
	if err != nil {
		log.Error("forceStart: ", err)
		return
	}
	defer resp.Body.Close()
	var payload GitBranchPaylod
	err = json.NewDecoder(resp.Body).Decode(&payload)

	commitHash := payload.Commit.Sha[0:7]

	var push Push
	go push.startPush(commitHash)
}

type GitPushPayload struct {
	Ref   string `json:"ref"`
	After string `json:"after"`
}

// listen for when we add a tag to the repo. when that happens replace the version in the DB where appropriate
// Github API for this is a bit lame
func gitPush(w http.ResponseWriter, r *http.Request) {
	log.Info("Git push :", r.Method)
	/*

		// dump request body
		var body []byte
		body, err := ioutil.ReadAll(r.Body)
		if err != nil {
			log.Info("Error reading request body: ", err)
		}
		log.Info("Request body: ", string(body))
	*/
	var payload GitPushPayload
	err := json.NewDecoder(r.Body).Decode(&payload)
	if err != nil {
		log.Info("Error decoding request body: ", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// make sure this is a push we care about
	if len(payload.After) > 7 {

		commitHash := payload.After[0:7]

		if payload.Ref == "refs/heads/"+gConfig.BRANCH_NAME {
			var push Push
			go push.startPush(commitHash) // start this in its own thread
			// this is an update to master
		} else if payload.Ref[0:9] == "refs/tags/" {
			newVersion := payload.Ref[9:]
			log.Info("Update version from: ", commitHash, " to: ", newVersion)
			gApp.database.UpdateVersion(commitHash, newVersion)
		} else {
			log.Info("Unknown ref: ", payload.Ref)
		}
	} else {
		log.Info("Not a push we care about ref:" + payload.Ref + " after:" + payload.After)
	}
}

func jobResult(w http.ResponseWriter, r *http.Request) {

	var result infra.ResultJob
	err := json.NewDecoder(r.Body).Decode(&result)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	gApp.processJobResult(&result)
}

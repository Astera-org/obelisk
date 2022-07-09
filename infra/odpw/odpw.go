package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"

	log "github.com/Astera-org/easylog"
	"github.com/Astera-org/obelisk/infra/gengo/infra"
)

var gConfig Config
var gApp OdpwApp
var VERSION string = "v0.1.0"

func main() {
	gConfig.Load()

	err := log.Init(
		log.SetLevel(log.INFO),
		log.SetFileName("odpw.log"),
	)
	if err != nil {
		panic(err)
	}

	gApp.Init()

	http.HandleFunc("/result", jobResult)
	http.HandleFunc("/push", gitPush)
	var serverAddr = fmt.Sprint(":", gConfig.SERVER_PORT)
	go http.ListenAndServe(serverAddr, nil)

	for true {
		var command string
		fmt.Scan(&command)
		switch command {
		case "q":
			os.Exit(0)
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
	fmt.Println("v: print version")
}

type GitPushPayload struct {
	ref   string
	after string
}

// listen for when we add a tag to the repo. when that happens replace the version in the DB where appropriate
// Github API for this is a bit lame
func gitPush(w http.ResponseWriter, r *http.Request) {

	var payload GitPushPayload
	err := json.NewDecoder(r.Body).Decode(&payload)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	commitHash := payload.after[0:7]

	if payload.ref == "refs/heads/master" {
		var push Push
		go push.startPush(commitHash) // start this in its own thread
		// this is an update to master
	} else if payload.ref[0:9] == "refs/tags/" {
		newVersion := payload.ref[9:]

		gApp.database.UpdateVersion(commitHash, newVersion)
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

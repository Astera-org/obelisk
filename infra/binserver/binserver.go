package main

import (
	"context"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"strconv"

	log "github.com/Astera-org/easylog"

	"goji.io/pat"
)

var gConfig Config
var defaultCtx = context.Background()
var gApp BinServerApp
var VERSION string = "v0.1.0"

func main() {
	gConfig.Load()

	err := log.Init(
		log.SetLevel(log.INFO),
		log.SetFileName("binserver.log"),
	)
	if err != nil {
		panic(err)
	}

	gApp.Init()

	log.Info("listening on ", gConfig.SERVER_PORT)

	fileServer := http.FileServer(http.Dir("./" + gConfig.BINARY_ROOT + "/"))
	mux := http.NewServeMux()
	mux.Handle("/binaries/", http.StripPrefix("/binaries", fileServer))

	//mux := goji.NewMux()
	//mux.HandleFunc(pat.Get("/getBinary/:id"), getBinary)
	mux.HandleFunc("/addBinary", addBinary)
	mux.HandleFunc("/completed/:id", handleCompleted)
	//mux.HandleFunc("/binaries/", getFile)

	go http.ListenAndServe(fmt.Sprint(":", gConfig.SERVER_PORT), mux)

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

func getFile(w http.ResponseWriter, r *http.Request) {
	log.Info("getFile: " + r.URL.Path)
	http.ServeFile(w, r, r.URL.Path)
}

func handleCompleted(w http.ResponseWriter, r *http.Request) {
	idStr := pat.Param(r, "id")
	jobID, err := strconv.Atoi(idStr)
	if err != nil {
		log.Info("invalid ID: ", idStr)
		http.Error(w, "invalid ID", http.StatusBadRequest)
		return
	}

	// see if <jobID>.zip already exists
	r.URL.Path = gConfig.COMPLETED_ROOT + "/" + idStr + ".zip"
	_, err = os.Stat(r.URL.Path)
	if err != nil {
		err = gApp.fetchRunResult(jobID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
	}

	http.ServeFile(w, r, r.URL.Path)
}

// LATER: how does this put this in the right place?
func addBinary(w http.ResponseWriter, r *http.Request) {
	// put the binary in a temp dir
	// hash the binary
	// make sure it isn't a duplicate
	// create new row for binary
	// move the temp binary to the right place in the fs
	// return the binary id

	// Parse our multipart form, 10 << 20 specifies a maximum
	// upload of 10 MB files.
	r.ParseMultipartForm(200 << 20)
	// FormFile returns the first file for the given key `myFile`
	// it also returns the FileHeader so we can get the Filename,
	// the Header and the size of the file
	file, fileHeader, err := r.FormFile("binary")
	if err != nil {
		log.Error("Error Uploading File: ", err)
		return
	}
	defer file.Close()

	log.Info("file header ", fileHeader.Header)
	log.Info("file size ", fileHeader.Size)
	log.Info("file name ", fileHeader.Filename)

	// ensure temp dir exists, otherwise tempFile fails
	dir, err := ioutil.TempDir("", gConfig.TEMP_DIR)
	if err != nil {
		log.Fatal(err)
	}

	// Create a temporary file within our temp directory that follows
	// a particular naming pattern
	tempFile, err := ioutil.TempFile(dir, "upload-*")
	if err != nil {
		log.Error(err)
		return
	}
	defer tempFile.Close()

	// read all of the contents of our uploaded file into a
	// byte array
	fileBytes, err := ioutil.ReadAll(file)
	if err != nil {
		log.Error(err)
		return
	}

	// write this byte array to our temporary file
	n, err := tempFile.Write(fileBytes)
	if err != nil {
		log.Error(err)
		return
	}
	log.Info("Temp file: ", tempFile.Name(), " bytes written ", n)

	// return that we have successfully uploaded our file!

	if gConfig.IS_LOCALHOST {
		w.Header().Set("Access-Control-Allow-Origin", "*")
	}

	fmt.Fprintf(w, "Successfully Uploaded File\n")

	// LATER:
	// file name
	// hash the binary
	// make sure it isn't a duplicate
	// create new row for binary
	// move the temp binary to the right place in the fs
	// return the binary id
}

func getBinary(w http.ResponseWriter, r *http.Request) {
	binID := pat.Param(r, "id")

	id, _ := strconv.Atoi(binID)

	name, version := gApp.db.getNameVersion(id)
	if name == "" {
		w.WriteHeader(http.StatusNotFound)
		return
	}
	r.URL.Path = "/" + name + "/" + version + "/binary"
	getFile(w, r)
}

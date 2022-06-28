package main

import (
	"context"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"

	"goji.io"
	"goji.io/pat"
)

var gConfig Config
var defaultCtx = context.Background()
var gDatabase Database
var VERSION string = "v0.1.0"

/*
- Connect to DB
- Create server
- Handle requests
*/

func main() {
	gConfig.Load()
	gDatabase.Connect()

	fmt.Println("listening on", gConfig.SERVER_PORT)

	mux := goji.NewMux()
	mux.HandleFunc(pat.Get("/getBinary/:id"), getBinary)
	mux.HandleFunc("/addBinary", addBinary)

	go http.ListenAndServe(":"+gConfig.SERVER_PORT, mux)

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
	file, handler, err := r.FormFile("binary")
	if err != nil {
		fmt.Println("Error Retrieving the File")
		fmt.Println(err)
		return
	}
	defer file.Close()
	fmt.Printf("Uploaded File: %+v\n", handler.Filename)
	fmt.Printf("File Size: %+v\n", handler.Size)
	fmt.Printf("MIME Header: %+v\n", handler.Header)

	// Create a temporary file within our temp-images directory that follows
	// a particular naming pattern
	tempFile, err := ioutil.TempFile(gConfig.TEMP_DIR, "upload-*")
	if err != nil {
		fmt.Println(err)
		return
	}
	defer tempFile.Close()

	// read all of the contents of our uploaded file into a
	// byte array
	fileBytes, err := ioutil.ReadAll(file)
	if err != nil {
		fmt.Println(err)
		return
	}
	// write this byte array to our temporary file
	tempFile.Write(fileBytes)
	// return that we have successfully uploaded our file!
	fmt.Fprintf(w, "Successfully Uploaded File\n")
	w.WriteHeader(http.StatusOK)
}

func getBinary(w http.ResponseWriter, r *http.Request) {
	binID := pat.Param(r, "id")

	name, version := gDatabase.getNameVersion(binID)
	if name == "" {
		w.WriteHeader(http.StatusNotFound)
		return
	}

}

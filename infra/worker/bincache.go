package main

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/Astera-org/obelisk/infra/gengo/infra"
	strip "github.com/grokify/html-strip-tags-go"
)

//map of binIDs to their names and versions
/*
Worker does the following:
-Get binID from Job
-Look up name/version from local cache
-Get name and version of unknown ID from jobczar
-Does name dir already exist?
-Create name dir
-Create name/data dir
-Get name/data/manifest.toml from binserver
-Get all files on the manifest
-Run all the files that are in the execute section
-Create name/version dir
-Get name/version/binary file from binserver
-Start work

*/

type BinCache struct {
	cache map[int32]*infra.BinInfo
}

func (bc *BinCache) Init() {
	bc.cache = make(map[int32]*infra.BinInfo)
}

// will download the binary if we don't have it locally
func (bc *BinCache) EnsureBinary(binID int32) *infra.BinInfo {
	if binID == 0 {
		return nil
	}

	if bc.cache[binID] == nil {
		// binary isn't in cache yet, so download it
		binInfo, err := downloadBinary(binID)
		if err != nil {
			return nil
		}

		bc.cache[binID] = binInfo
	}

	return bc.cache[binID]
}

func downloadBinary(binID int32) (*infra.BinInfo, error) {
	// fectch details from jobczar
	binInfo, err := gApp.jobCzar.GetBinInfo(gApp.context, binID)
	if err != nil {
		fmt.Println("error getting bin info:", err)
		return nil, err
	}

	if isBinaryLocal(binInfo) == false {

		dirName := gApp.rootDir + "/" + gConfig.BINDIR + "/" + binInfo.Name

		_, err = os.Stat(dirName)
		if err != nil {
			if os.IsNotExist(err) {
				// this dir doesn't exist
				err = startNewName(binInfo.Name)
				if err != nil {
					return nil, err
				}
			} else {
				// some other error
				fmt.Println("error1:", err)
				return nil, err
			}
		}
		err = downloadVersion(binInfo)
		if err != nil {
			fmt.Println("error downloading version:", err)
			return nil, err
		}
	}
	return binInfo, nil
}

func isBinaryLocal(binInfo *infra.BinInfo) bool {
	binaryPath := gApp.rootDir + "/" + gConfig.BINDIR + "/" + binInfo.Name + "/" + binInfo.Version + "/binary"
	_, err := os.Stat(binaryPath)
	if err != nil {
		fmt.Println("not local:", err)
		return false
	}
	return true
}

// download a new version of a binary
func downloadVersion(binInfo *infra.BinInfo) error {
	remoteDir := "binaries/" + binInfo.Name + "/" + binInfo.Version
	localDirName := gApp.rootDir + "/" + gConfig.BINDIR + "/" + binInfo.Name + "/" + binInfo.Version

	err := downloadDir(remoteDir, localDirName)
	if err != nil {
		fmt.Println("downloadVersion:", err)
		return nil
	}
	return nil
}

// we don't have this binary locally, so download it
func startNewName(binName string) error {

	localDirName := gApp.rootDir + "/" + gConfig.BINDIR + "/" + binName

	dataDirName := localDirName + "/data"

	remoteDir := "binaries/" + binName + "/data"
	err := downloadDir(remoteDir, dataDirName)
	if err != nil {
		fmt.Println("startNewName3:", err)
		return nil
	}

	//processManifest(remoteDir, dataDirName)
	return nil
}

func downloadDir(remoteDir string, localDirName string) error {

	err := os.MkdirAll(localDirName, 0755)
	if err != nil {
		fmt.Println("downloadDir:", localDirName, err)
		return err
	}

	url := "http://" + gConfig.BINSERVER_URL + "/" + remoteDir
	resp, err := http.Get(url)
	if err != nil {
		fmt.Println("error dl dir:", url)
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode == 404 {
		return errors.New("dir not found: " + url)
	}

	b, err := io.ReadAll(resp.Body)
	dirList := strip.StripTags(string(b))

	scanner := bufio.NewScanner(strings.NewReader(dirList))
	for scanner.Scan() {
		downloadFile(scanner.Text(), remoteDir, localDirName)
	}
	// TODO: write "ok" file
	return nil
}

func downloadFile(fileName string, remoteDir string, destDir string) error {
	fmt.Println("Download:" + fileName + " from " + remoteDir + " to " + destDir)
	destPath := remoteDir + "/" + fileName
	url := "http://" + gConfig.BINSERVER_URL + "/" + destPath
	resp, err := http.Get(url)
	if err != nil {
		fmt.Println("downloadFile1:", err)
		return err
	}

	defer resp.Body.Close()

	if resp.StatusCode == 404 {
		return errors.New("file not found: " + fileName)
	}

	localPath := destDir + "/" + fileName

	// Create the file
	out, err := os.Create(localPath)
	if err != nil {
		fmt.Println("downloadFile1:", err)
		return err
	}
	defer out.Close()

	// Write the body to file
	_, err = io.Copy(out, resp.Body)
	return err
}

/*

type Manifest struct {
	Download []string
	Execute  []string
}

func processManifest(remoteDir string, localDir string) {
	var manifest Manifest

	_, err := toml.DecodeFile(localDir+"/manifest.toml", &manifest)
	if err != nil {
		fmt.Println("error decoding manifest:", err)
		return
	}

	for _, fileName := range manifest.Download {
		err := downloadFile(fileName, remoteDir, localDir)
		if err != nil {
			fmt.Println("error downloading file:", fileName)
			return
		}
	}

	for _, fileName := range manifest.Execute {
		err := downloadFile(fileName, remoteDir, localDir)
		if err != nil {
			fmt.Println("error downloading file:", fileName)
			return
		}
	}

	for _, fileName := range manifest.Execute {
		exec.Command(fileName)
	}
}
*/

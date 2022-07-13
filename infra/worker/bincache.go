package main

import (
	"bufio"
	"errors"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"os/exec"
	"strings"

	log "github.com/Astera-org/easylog"

	commonInfra "github.com/Astera-org/obelisk/infra"
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
-Get all files in the package dir
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
	log.Info("downloadBinary: ", binID)
	// fetch bininfo from jobczar
	binInfo, err := gApp.jobCzar.GetBinInfo(gApp.context, binID)
	if err != nil {
		log.Error("error getting bin info:", err)
		return nil, err
	}

	// this will happen on restart of worker
	if isBinaryLocal(binInfo) == false {

		dirName := gApp.rootDir + "/" + gConfig.BINDIR + "/" + binInfo.Name

		_, err = os.Stat(dirName)
		if err != nil {
			if os.IsNotExist(err) {
				// this dir doesn't exist
				err = startNewName(binInfo.Name)
				if err != nil {
					os.RemoveAll(dirName)
					return nil, err
				}
			} else {
				// some other error
				log.Error("error1:", err)
				return nil, err
			}
		}
		err = downloadVersion(binInfo)
		if err != nil {
			log.Error("error downloading version:", err)
			// clean up dir
			os.RemoveAll(dirName + "/" + binInfo.Version)
			return nil, err
		}
		err = checkPackageHash(binInfo, dirName+"/"+binInfo.Version)
		if err != nil {
			log.Error("hash mismatch:", err)
			// clean up dir
			//TODO os.RemoveAll(dirName + "/" + binInfo.Version)
			return nil, err
		}
		writeOK(dirName)
	}
	return binInfo, nil
}

func writeOK(dirName string) {
	var file []byte = []byte{}
	_ = ioutil.WriteFile("ok", file, 0644)
}

// TODO: move this to commonInfra function
func checkPackageHash(binInfo *infra.BinInfo, dirName string) error {
	log.Info("checkPackageHash: ", dirName)
	list, err := os.ReadDir(dirName + "/package")
	if err != nil {
		log.Error("checkPackageHash:", err)
		return nil
	}

	var fileList []string = make([]string, len(list)+1)
	fileList[0] = binInfo.Name
	for n, dirItem := range list {
		fileList[n+1] = "package/" + dirItem.Name()
	}

	localHash, err := commonInfra.HashFileList(dirName, fileList)
	if err != nil {
		return err
	}
	if localHash != binInfo.PackageHash {
		return errors.New(localHash)
	}
	return nil
}

func isBinaryLocal(binInfo *infra.BinInfo) bool {
	binaryPath := gApp.rootDir + "/" + gConfig.BINDIR + "/" + binInfo.Name + "/" + binInfo.Version

	// check for "ok" file so we know the package was actually downloaded completely previously
	_, err := os.Stat(binaryPath + "/ok")
	if err != nil {
		log.Info("binary not local: ", err)
		// clean up dir
		os.RemoveAll(binaryPath)
		return false
	}

	return true
}

// we don't have this binary locally, so download it
func startNewName(binName string) error {
	log.Info("startNewName: ", binName)

	localDirName := gApp.rootDir + "/" + gConfig.BINDIR + "/" + binName

	dataDirName := localDirName + "/data"

	remoteDir := "binaries/" + binName + "/data"
	err := downloadDir(remoteDir, dataDirName)
	if err != nil {
		log.Error("startNewName3:", err)
		return nil
	}

	os.Chdir(dataDirName)
	// run setup.sh if it is there
	_, err = os.Stat("setup.sh")
	if err == nil {
		// TODO: make this executable
		exec.Command("setup.sh").Run()
	}

	return nil
}

func downloadDir(remoteDir string, localDirName string) error {
	log.Info("DD: ", remoteDir, " to: |", localDirName+"|")
	err := os.MkdirAll(localDirName, 0777)
	if err != nil {
		log.Error("downloadDir:", localDirName, err)
		return err
	}

	url := "http://" + gConfig.BINSERVER_URL + "/" + remoteDir
	log.Info("URL: ", url)
	resp, err := http.Get(url)
	if err != nil {
		log.Error("error dl dir:", url)
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode == 404 {
		return errors.New("dir not found: " + url)
	}

	b, err := io.ReadAll(resp.Body)
	dirList := strip.StripTags(string(b))
	//log.Info("dirList: ", dirList)

	scanner := bufio.NewScanner(strings.NewReader(dirList))
	for scanner.Scan() {
		// if scanner.Text() ends with /, then it is a dir
		if strings.HasSuffix(scanner.Text(), "/") {
			dirName := scanner.Text()[0 : len(scanner.Text())-1]
			err = downloadDir(remoteDir+"/"+dirName, localDirName+"/"+dirName)
		} else {
			downloadFile(scanner.Text(), remoteDir, localDirName)
		}
	}

	return nil
}

func downloadFile(fileName string, remoteDir string, destDir string) error {
	if fileName == "" {
		//log.Info("downloadFile: empty file name")
		return nil
	}
	log.Info("Download:" + fileName + " from " + remoteDir + " to " + destDir)
	destPath := remoteDir + "/" + fileName
	url := "http://" + gConfig.BINSERVER_URL + "/" + destPath
	resp, err := http.Get(url)
	if err != nil {
		log.Error("downloadFile1:", err)
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
		log.Error("downloadFile2:", err)
		return err
	}
	defer out.Close()

	// Write the body to file
	_, err = io.Copy(out, resp.Body)
	return err
}

// download a new version of a binary
func downloadVersion(binInfo *infra.BinInfo) error {
	remoteDir := "binaries/" + binInfo.Name + "/" + binInfo.Version
	localDirName := gApp.rootDir + "/" + gConfig.BINDIR + "/" + binInfo.Name + "/" + binInfo.Version

	err := downloadDir(remoteDir, localDirName)
	if err != nil {
		return err
	}
	if err := os.Chmod(localDirName+"/"+binInfo.Name, 0777); err != nil {
		return err
	}
	return nil
}

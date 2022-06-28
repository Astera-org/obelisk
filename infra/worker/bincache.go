package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"

	"github.com/Astera-org/obelisk/infra/gengo/infra"
	"github.com/BurntSushi/toml"
)

//map of binIDs to their names and versions
/*
Worker does the following:
Get binID from Job
Look up name/version from local cache
Get name and version of unknown ID from jobczar
Does name dir already exist?
Create name dir
Create name/data dir
Get name/data/manifest.toml from binserver
Get all files on the manifest
Run all the files that are in the execute section
Create name/version dir
Get name/version/binary file from binserver
Start work

*/

type BinCache struct {
	cache map[int32]*infra.BinInfo
}

func (bc *BinCache) GetBinInfo(binID int32) *infra.BinInfo {
	if bc.cache[binID] == nil {
		// fectch from jobczar
		binInfo, err := gApp.jobCzar.GetBinInfo(gApp.context, binID)
		if err != nil {
			return nil
		}

		dirName := fmt.Sprint(gConfig.BINDIR_ROOT, "/", binInfo.Name)

		err = os.Chdir(dirName)
		if err != nil {
			// first time we have gotten this binary name
			err = os.Mkdir(dirName, 0755)
			if err != nil {
				fmt.Println("error creating dir:", dirName)
				return nil
			}
			dataDirName := fmt.Sprint(dirName, "/data")
			err = os.Mkdir(dataDirName, 0755)
			if err != nil {
				fmt.Println("error creating dir:", dataDirName)
				return nil
			}
			err = os.Chdir(dataDirName)
			if err != nil {
				fmt.Println("error changing dir:", dataDirName)
				return nil
			}
			err = downloadFile("manifest.toml")
			if err != nil {
				fmt.Println("error downloading file:", "manifest.toml")
				return nil
			}

			processManifest(dataDirName)
			binDirName := fmt.Sprint(dirName, "/", binInfo.Name)
			err = os.Mkdir(dataDirName, 0755)
			if err != nil {
				fmt.Println("error creating dir:", dataDirName)
				return nil
			}
			downloadFile("binary", binDirName)

			bc.cache[binID] = binInfo

		}

	}

	return bc.cache[binID]
}

func downloadFile(fileName string, destDir string) error {
	fmt.Println("fetching file:", fileName)

	destFile := destDir + "/" + fileName

	resp, err := http.Get(gConfig.BINSERVER_URL + "/" + fileName)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	// Create the file
	out, err := os.Create(destFile)
	if err != nil {
		return err
	}
	defer out.Close()

	// Write the body to file
	_, err = io.Copy(out, resp.Body)
	return err
}

type Manifest struct {
	Download []string
	Execute  []string
}

func processManifest(dataDir string) {
	fmt.Println("processing manifest:", dataDir)
	var manifest Manifest

	_, err := toml.DecodeFile("manifest.toml", &manifest)
	if err != nil {
		fmt.Println("error decoding manifest:", err)
		return
	}

	for _, fileName := range manifest.Download {
		err := downloadFile(fileName, dataDir)
		if err != nil {
			fmt.Println("error downloading file:", fileName)
			return
		}
	}

	for _, fileName := range manifest.Execute {
		err := downloadFile(fileName, dataDir)
		if err != nil {
			fmt.Println("error downloading file:", fileName)
			return
		}
	}

	for _, fileName := range manifest.Execute {
		exec.Command(fileName)
	}

}

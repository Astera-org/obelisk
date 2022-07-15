package commonInfra

// Utility functions to work with the infra system

import (
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"strings"

	log "github.com/Astera-org/easylog"
	"github.com/Astera-org/obelisk/infra/gengo/infra"
)

func WriteResults(score float64, steps int32, seconds int32) {

	result := infra.ResultJob{
		Seconds: seconds,
		Steps:   steps,
		Score:   score,
	}

	file, _ := json.MarshalIndent(result, "", " ")

	_ = ioutil.WriteFile("result.json", file, 0644)
}

func CalculatePackageHash(binName string, dirName string) (string, error) {
	log.Info("calculatePackageHash: ", binName, " ", dirName)
	list, _ := os.ReadDir(dirName + "/package")
	// ok if package doesn't exist

	var fileList []string = make([]string, len(list)+1)
	fileList[0] = binName
	for n, dirItem := range list {
		fileList[n+1] = "package/" + dirItem.Name()
	}

	hash, err := HashFileList(dirName, fileList)
	if err != nil {
		return "", err
	}

	return hash, nil
}

// calculates the hash of the given list of files. All should be relative to the root path
func HashFileList(rootPath string, fileList []string) (string, error) {

	hasher := sha1.New()
	for _, fileName := range fileList {
		file, err := os.Open(rootPath + "/" + fileName)
		if err != nil {
			return "", err
		}
		defer file.Close()

		if _, err := io.Copy(hasher, file); err != nil {
			return "", err
		}
	}

	return hex.EncodeToString(hasher.Sum(nil)), nil
}

func RunCommand(command string) error {
	args := strings.Fields(command)
	return exec.Command(args[0], args[1:]...).Run()
}

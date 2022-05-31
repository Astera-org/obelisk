package sim

import (
	"encoding/json"
	"flag"
	"fmt"
	"github.com/emer/emergent/etime"
	"io/ioutil"
	"os"

	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/goki/gi/gi"
)

type CmdArgs struct {
	SaveWts        bool             `desc:"for command-line run only, auto-save final weights after each run"`
	NoGui          bool             `desc:"if true, runing in no GUI mode"`
	RandomizeEvery bool             `desc:"If true, randomize seed for every run"`
	LogSetParams   bool             `desc:"if true, print message for all params that are set"`
	NeedsNewRun    bool             `desc:"flag to initialize NewRun if last one finished"`
	RndSeeds       []int64          `desc:"a list of random seeds to use for each run"`
	NetData        *netview.NetData `desc:"net data for recording in nogui mode"`

	saveEpcLog   bool
	saveRunLog   bool
	saveTrialLog bool // Test Trial
	saveNetData  bool
	note         string
	hyperFile    string
	paramsFile   string

	NoRun        bool `desc:"If true, don't run at all.'"`
	MaxRuns      int  `desc:"maximum number of model runs to perform (starting from StartRun)"`
	MaxEpcs      int  `desc:"maximum number of epochs to run per model run"`
	StartRun     int  `desc:"starting run number -- typically 0 but can be set in command args for parallel runs on a cluster"`
	PreTrainEpcs int  `desc:"number of epochs to run for pretraining"`
}

// ParseArgs updates the Sim object with command line arguments.
func (ss *Sim) ParseArgs() {
	flag.StringVar(&ss.Params.ExtraSets, "params", "", "ParamSet name to use -- must be valid name as listed in compiled-in params or loaded params")
	flag.StringVar(&ss.Tag, "tag", "", "extra tag to add to file names saved from this run")
	flag.IntVar(&ss.CmdArgs.StartRun, "run", 0, "starting run number -- determines the random seed -- runs counts from there -- can do all runs in parallel by launching separate jobs with each run, runs = 1")
	flag.IntVar(&ss.CmdArgs.MaxRuns, "runs", 10, "number of runs to do (note that MaxEpcs is in paramset)")
	flag.IntVar(&ss.CmdArgs.MaxEpcs, "epochs", 150, "number of epochs per run")
	flag.IntVar(&ss.CmdArgs.PreTrainEpcs, "pretrainEpochs", 150, "number of epochs to run for pretraining")
	flag.BoolVar(&ss.CmdArgs.LogSetParams, "setparams", false, "if true, print a record of each parameter that is set")
	flag.BoolVar(&ss.CmdArgs.RandomizeEvery, "randomize", false, "If true, randomize seed for every run")
	flag.BoolVar(&ss.CmdArgs.SaveWts, "wts", false, "if true, save final weights after each run")
	flag.StringVar(&ss.CmdArgs.note, "note", "", "user note -- describe the run params etc")
	flag.BoolVar(&ss.CmdArgs.saveEpcLog, "epclog", true, "if true, save train epoch log to file")
	flag.BoolVar(&ss.CmdArgs.saveTrialLog, "triallog", true, "if true, save test trial log to file. May be large.")
	flag.BoolVar(&ss.CmdArgs.saveRunLog, "runlog", true, "if true, save run log to file")
	flag.BoolVar(&ss.CmdArgs.saveNetData, "netdata", false, "if true, save network activation etc data from testing trials, for later viewing in netview")
	flag.BoolVar(&ss.CmdArgs.NoGui, "nogui", len(os.Args) > 1, "if not passing any other args and want to run nogui, use nogui")
	flag.StringVar(&ss.CmdArgs.hyperFile, "hyperFile", "", "Name of the file to output hyperparameter data. If not empty string, program should write and then exit")
	flag.StringVar(&ss.CmdArgs.paramsFile, "paramsFile", "", "Name of the file to input parameters from.")
	flag.Parse()
	// TODO reformat jsonl to be strings only, causing a read issue
	if ss.CmdArgs.hyperFile != "" {
		file, _ := json.MarshalIndent(ss.Params.Params, "", "  ")
		_ = ioutil.WriteFile(ss.CmdArgs.hyperFile, file, 0644)
		ss.CmdArgs.NoRun = true
		ss.CmdArgs.NoGui = true
		return
	}

	// ToDO from michael to andrew - need to account for paramsFile and extrasheet not always being the same thing - should be separate parameter imo
	if ss.CmdArgs.paramsFile != "" {
		ss.ApplyHyperFromCMD(ss.CmdArgs.paramsFile)
	}

	if ss.CmdArgs.note != "" {
		fmt.Printf("note: %s\n", ss.CmdArgs.note)
	}
	if ss.Params.ExtraSets != "" {
		fmt.Printf("Using ParamSet: %s\n", ss.Params.ExtraSets)
	}
	ss.CmdArgs.RndSeeds = make([]int64, 100) // make enough for plenty of runs
	for i := 0; i < len(ss.CmdArgs.RndSeeds); i++ {
		ss.CmdArgs.RndSeeds[i] = int64(i) + 1 // exclude 0
	}
	if ss.CmdArgs.RandomizeEvery {
		ss.NewRndSeed()
	}

	if ss.CmdArgs.saveNetData {
		ss.CmdArgs.NetData = &netview.NetData{}
		ss.CmdArgs.NetData.Init(ss.Net, 200) // 200 = amount to save
	}
	if ss.CmdArgs.SaveWts {
		fmt.Printf("Saving final weights per run\n")
	}
}

func (ss *Sim) ApplyHyperFromCMD(path string) {
	ss.CmdArgs.paramsFile = path
	jsonFile, err := os.Open(ss.CmdArgs.paramsFile)
	if err != nil {
		fmt.Println("Params file error: " + err.Error())
		return
	}
	defer jsonFile.Close()
	byteValue, _ := ioutil.ReadAll(jsonFile)
	loadedParams := params.Sets{}
	json.Unmarshal(byteValue, &loadedParams)
	if len(loadedParams) == 0 {
		fmt.Println("Unable to load parameters from file: " + ss.CmdArgs.paramsFile)
		return
	}
	ss.Params.ExtraSets = loadedParams[0].Name // just make them the same
	ss.Params.Params = append(ss.Params.Params, loadedParams[0])
}

// RunFromArgs uses command line arguments to run the model.
func (ss *Sim) RunFromArgs() {
	if ss.CmdArgs.NoRun {
		return
	}
	ss.Init()

	fmt.Printf("Running %d Runs starting at %d\n", ss.CmdArgs.MaxRuns, ss.CmdArgs.StartRun)
	ss.Run.Set(ss.CmdArgs.StartRun)
	ss.Run.Max = ss.CmdArgs.StartRun + ss.CmdArgs.MaxRuns
	ss.NewRun()
	ss.Train(etime.TimesN) // Train all Runs

	ss.Logs.CloseLogFiles()

	if ss.CmdArgs.saveNetData {
		ndfn := ss.Net.Nm + "_" + ss.RunName() + ".netdata.gz"
		ss.CmdArgs.NetData.SaveJSON(gi.FileName(ndfn))
	}
}

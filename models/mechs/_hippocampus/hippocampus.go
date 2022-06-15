// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// one2many is a copy of ra25, but instead of a single output
// associated with each input, there are multiple. The Correl
// metric that's reported is computed based on correlation with
// the closest found pattern.

package main

import (
	"fmt"
	"github.com/Astera-org/models/library/common"
	"github.com/Astera-org/models/library/sim"
	"github.com/emer/axon/axon"
	"github.com/emer/axon/hip"
	"github.com/emer/emergent/egui"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/patgen"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/etable/etable"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
	"log"
)

var ProgramName = "Hippocampus"

var TestEnvHip = EnvHip{IsTest: true}
var TrainEnvHip = EnvHip{}

// TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
// accum is true.  Note that we're accumulating stats here on the Sim side so the
// core algorithm side remains as simple as possible, and doesn't need to worry about
// different time-scales over which stats could be accumulated etc.
// You can also aggregate directly from log data, as is done for testing stats
func TrialStats(ss *sim.Sim, accum bool) {
	outLay := ss.Net.LayerByName("ECout").(axon.AxonLayer).AsAxon()
	ss.Stats.SetFloat("TrlCosDiff", float64(outLay.CosDiff.Cos))
	ss.Stats.SetFloat("TrlUnitErr", outLay.PctUnitErr())
	if accum {
		ss.Stats.SetFloat("SumUnitErr", ss.Stats.Float("SumUnitErr")+ss.Stats.Float("TrlUnitErr"))
		ss.Stats.SetFloat("SumCosDiff", ss.Stats.Float("SumCosDiff")+ss.Stats.Float("TrlCosDiff"))
		if ss.Stats.Float("TrlCosDiff") != 0 {
			ss.Stats.SetInt("CntErr", ss.Stats.Int("CntErr")+1)
		}
	}
}

type HipSim struct {
	sim.Sim
	// Specific to the one2many module
	Hip       HipParams    `desc:"hippocampus sizing parameters"`
	PoolVocab patgen.Vocab `view:"no-inline" desc:"pool patterns vocabulary"`
	Pat       PatParams
}

func (ss *HipSim) New() {
	ss.Sim.New()
	ss.PoolVocab = patgen.Vocab{}
	ss.Hip = HipParams{}
	ss.Hip.Defaults()
	ss.Pat.Defaults()
	ss.Hip.Defaults()
	ss.Pat.Defaults()
	ss.Run.Cur = 0 // for initializing envs if using Gui // TODO Makesure this works right. It was StartRun.
	ss.Update()
}

func (ss *HipSim) Update() {
	ss.Hip.Update()
}

func main() {
	// TheSim is the overall state for this simulation
	var TheSim HipSim
	TheSim.New()
	Config(&TheSim)

	if TheSim.CmdArgs.NoGui {
		if !TheSim.CmdArgs.NoRun {
			// TODO This might not be the behavior we want! PreTrain needs to be refactored, though.
			//PreTrain(&TheSim.Sim)
		}
		TheSim.RunFromArgs()
	} else {
		gimain.Main(func() { // this starts gui -- requires valid OpenGL display connection (e.g., X11)
			window := TheSim.ConfigGui(ProgramName, "Hippocampus AB-AC", `This demonstrates a basic Hippocampus model in Axon. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)
			ConfigGui(&TheSim)
			sim.GuiRun(&TheSim.Sim, window)
			PreTrain(&TheSim.Sim)
		})
	}
}

func OpenPat(dt *etable.Table, fname, name, desc string) {
	err := dt.OpenCSV(gi.FileName(fname), etable.Tab)
	if err != nil {
		log.Println(err)
		return
	}
	dt.SetMetaData("name", name)
	dt.SetMetaData("desc", desc)
}

func OpenFixedPatterns(ss *HipSim) {
	OpenPat(TrainEnvHip.EvalTables[TrainAB], "hippoinputs/trainab.tsv", "TrainAB", "")
	OpenPat(TrainEnvHip.EvalTables[TrainAC], "hippoinputs/trainac.tsv", "TrainAC", "")
	OpenPat(TrainEnvHip.EvalTables[TrainAll], "hippoinputs/trainall.tsv", "TrainAll", "")

	OpenPat(TestEnvHip.EvalTables[TestAB], "hippoinputs/testab.tsv", "TestAB", "")
	OpenPat(TestEnvHip.EvalTables[TestAC], "hippoinputs/testac.tsv", "TestAC", "")
	OpenPat(TestEnvHip.EvalTables[TestLure], "hippoinputs/testlure.tsv", "TestLure", "")

}

// Config configures all the elements using the standard functions
func Config(ss *HipSim) {
	// These need to be initialized before ConfigPats
	TrainEnvHip.InitTables(TrainAB, TrainAC, PretrainLure, TrainAll)
	TestEnvHip.InitTables(TestAB, TestAC, TestLure)

	ConfigPats(ss)
	//OpenFixedPatterns(ss) //todo ths is for debugging, shoudl be removed later

	//ss.Initialization = func() { // TODO Why? Was this necessary? Reconfiguring stuff messes it up.
	//	ss.InitRndSeed()
	//	ReconfigPatsAndNet(ss)
	//	ConfigEnv(ss) // re-config env just in case a different set of patterns was
	//	// selected or patterns have been modified etc
	//	//OpenFixedPatterns(ss) //todo should be removed, htis is for debugging purposes
	//}

	ConfigParams(&ss.Sim)
	// Parse arguments before configuring the network and env, in case parameters are set.
	ss.ParseArgs()
	ConfigEnv(ss)

	ss.TestInterval = 1
	ConfigNet(ss, ss.Net)
	InitHipStats(&ss.Sim)
	ss.ConfigLogItems()
	ConfigHipItems(&ss.Sim)
	ss.ConfigLogs()
	common.AddDefaultTrainCallbacks(&ss.Sim)
	AddHipCallbacks(ss)

	conditionEnvs := TrainEnvHip.AddTaskSwitching(&ss.Sim)
	ss.Trainer.Callbacks = append(ss.Trainer.Callbacks, *conditionEnvs)
}

func ConfigGui(ss *HipSim) {
	// TODO Add a separator to put this in its own section.
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "PreTrain",
		Icon:    "fast-fwd",
		Tooltip: "Does full pretraining.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				ss.GUI.IsRunning = true
				ss.GUI.StopNow = false
				go PreTrain(&ss.Sim)
			}
		}})

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "PreTrain Step",
		Icon:    "step-fwd",
		Tooltip: "One trial of pretraining.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				ss.GUI.IsRunning = true
				ss.GUI.StopNow = false
				go PreTrainTrial(&ss.Sim)
				ss.GUI.Stopped()
			}
		}})

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Env",
		Icon:    "gear",
		Tooltip: "select training input patterns: AB or AC.",
		Active:  egui.ActiveStopped,
		Func:    func() {}, // TODO Call SetEnv
	})

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Rebuild Net",
		Icon:    "reset",
		Tooltip: "Rebuild network with current params.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ReconfigPatsAndNet(ss)
		},
	})

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Run Stats",
		Icon:    "file-data",
		Tooltip: "Compute stats from run log -- avail in plot.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.LogRunStats()
		},
	})
}

//////////////////////////////////////////////////////////////////////////////////////////////
//// 		Configs

func ConfigEnv(ss *HipSim) {
	ss.TestEnv = &TestEnvHip
	ss.TrainEnv = &TrainEnvHip
	TestEnvHip.TrainEnv = &TrainEnvHip
	ss.CmdArgs.PreTrainEpcs = 10    //from hip sim
	ss.Stats.SetInt("NZeroStop", 1) //TODO move this, should be a command line args
	ss.TrialStatsFunc = TrialStats

	// TODO PCA seems to hang in internal Dlatrd function for hippocampus.
	ss.PCAInterval = -1

	TrainEnvHip.Nm = "TrainEnv"
	TrainEnvHip.Dsc = "training params and state"

	TrainEnvHip.Table = etable.NewIdxView(TrainEnvHip.EvalTables[TrainAB])
	TrainEnvHip.CurrentTableName = string(TrainAB)
	// to simulate training items in order, uncomment this line:
	// ss.TrainEnv.Sequential = true
	TrainEnvHip.SetSequential(true) //todo this should be removed, this is done to compare between original and old
	TrainEnvHip.Validate()
	ss.Run.Max = ss.CmdArgs.MaxRuns
	ss.Run.Cur = ss.CmdArgs.StartRun
	TrainEnvHip.Epoch().Max = ss.CmdArgs.MaxEpcs

	// MaxEpcs is consulted for early stopping in TrainTrial and is split between AB and AC

	TestEnvHip.Nm = "TestEnv"
	TestEnvHip.Dsc = "testing params and state"
	TestEnvHip.Table = etable.NewIdxView(TestEnvHip.EvalTables[TestAB])
	TestEnvHip.CurrentTableName = string(TestAB)
	TestEnvHip.SetSequential(true)
	ss.TestEnv.Validate()

	ss.TrainEnv.Init(ss.CmdArgs.StartRun)
	ss.TrainEnv.Trial().Cur = 0
	ss.TestEnv.Init(ss.CmdArgs.StartRun)
	ss.TestEnv.Trial().Cur = 0
}

//ConfigPats used to configure patterns
func ConfigPats(ss *HipSim) {

	trainEnv := &TrainEnvHip
	testEnv := &TestEnvHip
	hp := &ss.Hip
	ecY := hp.ECSize.Y
	ecX := hp.ECSize.X
	plY := hp.ECPool.Y // good idea to get shorter vars when used frequently
	plX := hp.ECPool.X // makes much more readable
	npats := ss.Pat.ListSize
	pctAct := hp.ECPctAct
	minDiff := ss.Pat.MinDiffPct
	nOn := patgen.NFmPct(pctAct, plY*plX)
	ctxtflip := patgen.NFmPct(ss.Pat.CtxtFlipPct, nOn)
	patgen.AddVocabEmpty(ss.PoolVocab, "empty", npats, plY, plX)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "A", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "B", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "C", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "lA", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "lB", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "ctxt", 3, plY, plX, pctAct, minDiff) // totally diff

	for i := 0; i < (ecY-1)*ecX*3; i++ { // 12 contexts! 1: 1 row of stimuli pats; 3: 3 diff ctxt bases
		list := i / ((ecY - 1) * ecX)
		ctxtNm := fmt.Sprintf("ctxt%d", i+1)
		tsr, _ := patgen.AddVocabRepeat(ss.PoolVocab, ctxtNm, npats, "ctxt", list)
		patgen.FlipBitsRows(tsr, ctxtflip, ctxtflip, 1, 0)
		//todo: also support drifting
		//solution 2: drift based on last trial (will require sequential learning)
		//patgen.VocabDrift(ss.PoolVocab, ss.NFlipBits, "ctxt"+strconv.Itoa(i+1))
	}

	trainAB, testAB := trainEnv.EvalTables[TrainAB], testEnv.EvalTables[TestAB]
	trainAC, testAC := trainEnv.EvalTables[TrainAC], testEnv.EvalTables[TestAC]
	preTrainLure, testLure := trainEnv.EvalTables[PretrainLure], testEnv.EvalTables[TestLure]
	trainAll := trainEnv.EvalTables[TrainAll]

	patgen.InitPats(trainAB, "TrainAB", "TrainAB Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(trainAB, ss.PoolVocab, "Input", []string{"A", "B", "ctxt1", "ctxt2", "ctxt3", "ctxt4"})
	patgen.MixPats(trainAB, ss.PoolVocab, "ECout", []string{"A", "B", "ctxt1", "ctxt2", "ctxt3", "ctxt4"})

	patgen.InitPats(testAB, "TestAB", "TestAB Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(testAB, ss.PoolVocab, "Input", []string{"A", "empty", "ctxt1", "ctxt2", "ctxt3", "ctxt4"})
	patgen.MixPats(testAB, ss.PoolVocab, "ECout", []string{"A", "B", "ctxt1", "ctxt2", "ctxt3", "ctxt4"})

	patgen.InitPats(trainAC, "TrainAC", "TrainAC Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(trainAC, ss.PoolVocab, "Input", []string{"A", "C", "ctxt5", "ctxt6", "ctxt7", "ctxt8"})
	patgen.MixPats(trainAC, ss.PoolVocab, "ECout", []string{"A", "C", "ctxt5", "ctxt6", "ctxt7", "ctxt8"})

	patgen.InitPats(testAC, "TestAC", "TestAC Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(testAC, ss.PoolVocab, "Input", []string{"A", "empty", "ctxt5", "ctxt6", "ctxt7", "ctxt8"})
	patgen.MixPats(testAC, ss.PoolVocab, "ECout", []string{"A", "C", "ctxt5", "ctxt6", "ctxt7", "ctxt8"})

	patgen.InitPats(preTrainLure, "PreTrainLure", "PreTrainLure Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(preTrainLure, ss.PoolVocab, "Input", []string{"lA", "lB", "ctxt9", "ctxt10", "ctxt11", "ctxt12"}) // arbitrary ctxt here
	patgen.MixPats(preTrainLure, ss.PoolVocab, "ECout", []string{"lA", "lB", "ctxt9", "ctxt10", "ctxt11", "ctxt12"}) // arbitrary ctxt here

	patgen.InitPats(testLure, "TestLure", "TestLure Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(testLure, ss.PoolVocab, "Input", []string{"lA", "empty", "ctxt9", "ctxt10", "ctxt11", "ctxt12"}) // arbitrary ctxt here
	patgen.MixPats(testLure, ss.PoolVocab, "ECout", []string{"lA", "lB", "ctxt9", "ctxt10", "ctxt11", "ctxt12"})    // arbitrary ctxt here

	//shuold potentially go in Environments
	trainAll = trainAB.Clone()
	trainAll.AppendRows(trainAC)
	trainAll.AppendRows(preTrainLure)
	trainEnv.EvalTables[TrainAll] = trainAll

	//trainEnv.EvalTables[HipTableTypes("TrainAB")].SaveCSV("TrainAB_Sample.csv", etable.Comma, true)
}

func (ss *HipSim) OpenPat(dt *etable.Table, fname, name, desc string) {
	err := dt.OpenCSV(gi.FileName(fname), etable.Tab)
	if err != nil {
		log.Println(err)
		return
	}
	dt.SetMetaData("name", name)
	dt.SetMetaData("desc", desc)
}

func ConfigNet(ss *HipSim, net *axon.Network) {
	net.InitName(net, ProgramName)
	hp := &ss.Hip

	in := net.AddLayer4D("Input", hp.ECSize.Y, hp.ECSize.X, hp.ECPool.Y, hp.ECPool.X, emer.Input)
	ecin := net.AddLayer4D("ECin", hp.ECSize.Y, hp.ECSize.X, hp.ECPool.Y, hp.ECPool.X, emer.Hidden)
	ecout := net.AddLayer4D("ECout", hp.ECSize.Y, hp.ECSize.X, hp.ECPool.Y, hp.ECPool.X, emer.Target) // clamped in plus phase
	ca1 := net.AddLayer4D("CA1", hp.ECSize.Y, hp.ECSize.X, hp.CA1Pool.Y, hp.CA1Pool.X, emer.Hidden)
	dg := net.AddLayer2D("DG", hp.DGSize.Y, hp.DGSize.X, emer.Hidden)
	ca3 := net.AddLayer2D("CA3", hp.CA3Size.Y, hp.CA3Size.X, emer.Hidden)

	ecin.SetClass("EC")
	ecout.SetClass("EC")

	ecin.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Input", YAlign: relpos.Front, Space: 2})
	ecout.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "ECin", YAlign: relpos.Front, Space: 2})
	dg.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Input", YAlign: relpos.Front, XAlign: relpos.Left, Space: 0})
	ca3.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "DG", YAlign: relpos.Front, XAlign: relpos.Left, Space: 0})
	ca1.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "CA3", YAlign: relpos.Front, Space: 2})

	onetoone := prjn.NewOneToOne()
	pool1to1 := prjn.NewPoolOneToOne()
	full := prjn.NewFull()

	net.ConnectLayers(in, ecin, onetoone, emer.Forward)
	net.ConnectLayers(ecout, ecin, onetoone, emer.Back)

	// EC <-> CA1 encoder pathways
	if false { // false = actually works better to use the regular projections here
		pj := net.ConnectLayersPrjn(ecin, ca1, pool1to1, emer.Forward, &hip.EcCa1Prjn{})
		pj.SetClass("EcCa1Prjn")
		pj = net.ConnectLayersPrjn(ca1, ecout, pool1to1, emer.Forward, &hip.EcCa1Prjn{})
		pj.SetClass("EcCa1Prjn")
		pj = net.ConnectLayersPrjn(ecout, ca1, pool1to1, emer.Back, &hip.EcCa1Prjn{})
		pj.SetClass("EcCa1Prjn")
	} else {
		pj := net.ConnectLayers(ecin, ca1, pool1to1, emer.Forward)
		pj.SetClass("EcCa1Prjn")
		pj = net.ConnectLayers(ca1, ecout, pool1to1, emer.Forward)
		pj.SetClass("EcCa1Prjn")
		pj = net.ConnectLayers(ecout, ca1, pool1to1, emer.Back)
		pj.SetClass("EcCa1Prjn")
	}

	// Perforant pathway
	ppathDG := prjn.NewUnifRnd()
	ppathDG.PCon = hp.DGPCon
	ppathCA3 := prjn.NewUnifRnd()
	ppathCA3.PCon = hp.CA3PCon

	pj := net.ConnectLayersPrjn(ecin, dg, ppathDG, emer.Forward, &hip.CHLPrjn{})
	pj.SetClass("HippoCHL")

	if true { // toggle for bcm vs. ppath, zycyc: must use false for orig_param, true for def_param
		pj = net.ConnectLayersPrjn(ecin, ca3, ppathCA3, emer.Forward, &hip.EcCa1Prjn{})
		pj.SetClass("PPath")
		pj = net.ConnectLayersPrjn(ca3, ca3, full, emer.Lateral, &hip.EcCa1Prjn{})
		pj.SetClass("PPath")
	} else {
		// so far, this is sig worse, even with error-driven MinusQ1 case (which is better than off)
		pj = net.ConnectLayersPrjn(ecin, ca3, ppathCA3, emer.Forward, &hip.CHLPrjn{})
		pj.SetClass("HippoCHL")
		pj = net.ConnectLayersPrjn(ca3, ca3, full, emer.Lateral, &hip.CHLPrjn{})
		pj.SetClass("HippoCHL")
	}

	// always use this for now:
	if false { // todo: was true
		pj = net.ConnectLayersPrjn(ca3, ca1, full, emer.Forward, &hip.CHLPrjn{})
		pj.SetClass("HippoCHL")
	} else {
		// note: this requires lrate = 1.0 or maybe 1.2, doesn't work *nearly* as well
		pj = net.ConnectLayers(ca3, ca1, full, emer.Forward) // default con
		// pj.SetClass("HippoCHL")
	}

	// Mossy fibers
	mossy := prjn.NewUnifRnd()
	mossy.PCon = hp.MossyPCon
	pj = net.ConnectLayersPrjn(dg, ca3, mossy, emer.Forward, &hip.CHLPrjn{}) // no learning
	pj.SetClass("HippoCHL")

	// using 4 threads total (rest on 0)
	dg.SetThread(1)
	ca3.SetThread(2)
	ca1.SetThread(3) // this has the most

	// note: if you wanted to change a layer type from e.g., Target to Compare, do this:
	// outLay.SetType(emer.Compare)
	// that would mean that the output layer doesn't reflect target values in plus phase
	// and thus removes error-driven learning -- but stats are still computed.

	net.Defaults()

	ss.Params.SetObject("Network")
	//ss.SetParams("Network", ss.LogSetParams) // only set Network params
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	net.InitWts()
}

// SetParamsSet sets the params for given params.Set name.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func SetParamsSet(ss *HipSim, setNm string, sheet string, setMsg bool) error {
	pset, err := ss.Params.Params.SetByNameTry(setNm)
	if err != nil {
		return err
	}
	if sheet == "" || sheet == "Network" {
		netp, ok := pset.Sheets["Network"]
		if ok {
			ss.Net.ApplyParams(netp, setMsg)
		}
	}

	if sheet == "" || sheet == "Sim" {
		simp, ok := pset.Sheets["Sim"]
		if ok {
			simp.Apply(ss, setMsg)
		}
	}

	if sheet == "" || sheet == "Hip" {
		simp, ok := pset.Sheets["Hip"]
		if ok {
			simp.Apply(&ss.Hip, setMsg)
		}
	}

	if sheet == "" || sheet == "Pat" {
		simp, ok := pset.Sheets["Pat"]
		if ok {
			simp.Apply(&ss.Pat, setMsg)
		}
	}

	// note: if you have more complex environments with parameters, definitely add
	// sheets for them, e.g., "TrainEnv", "TestEnv" etc
	return err
}

var KiT_HipSim = kit.Types.AddType(&HipSim{}, SimProps)
var SimProps = ki.Props{
	"CallMethods": ki.PropSlice{
		{"SaveWeights", ki.Props{
			"desc": "save network weights to file",
			"icon": "file-save",
			"Args": ki.PropSlice{
				{"File Name", ki.Props{
					"ext": ".wts,.wts.gz",
				}},
			},
		}},
		{"SetEnv", ki.Props{
			"desc": "select which set of patterns to train on: AB or AC",
			"icon": "gear",
			"Args": ki.PropSlice{
				{"Train on AC", ki.Props{}},
			},
		}},
	},
}

// zycyc
// ModelSizeParams are the parameters to run for outer crossed factor testing
//var ModelSizeParams = []string{"BigHip"}
// var ModelSizeParams = []string{"MedHip", "BigHip"}
var ModelSizeParams = []string{"MedHip"}

// ListSizeParams are the parameters to run for inner crossed factor testing
// var ListSizeParams = []string{"List010"}

var ListSizeParams = []string{"List040", "List060"}

// var ListSizeParams = []string{"List020", "List040", "List060", "List080", "List100"}

// TwoFactorRun runs outer-loop crossed with inner-loop params
// TODO This needs to be called
func TwoFactorRun(ss *HipSim) {
	tag := ss.Tag
	usetag := tag
	if usetag != "" {
		usetag += "_"
	}
	for _, modelSize := range ModelSizeParams {
		for _, listSize := range ListSizeParams {
			ss.Tag = usetag + modelSize + "_" + listSize
			ss.InitRndSeed()
			SetParamsSet(ss, modelSize, "", ss.CmdArgs.LogSetParams)
			SetParamsSet(ss, listSize, "", ss.CmdArgs.LogSetParams)
			ReconfigPatsAndNet(ss) // note: this applies Base params to Network
			//ConfigEnv(ss)
			ss.GUI.StopNow = false
			PreTrain(&ss.Sim) // zycyc
			ss.NewRun()
			ss.Train(etime.TimesN)
		}
	}
	ss.Tag = tag
}

func ReconfigPatsAndNet(ss *HipSim) {
	ss.Update()
	ConfigPats(ss)
	ss.Net = &axon.Network{} // start over with new network
	ConfigParams(&ss.Sim)    // Make sure we've got the right params. // TODO This will clobber any param changes made in GUI
	ConfigNet(ss, ss.Net)
	if ss.GUI.NetView != nil {
		ss.GUI.NetView.SetNet(ss.Net)
		ss.GUI.NetView.Update() // issue #41 closed
	}
}

// Callbacks related

// testEpochHip is copied from looper.go's loopEpoch, except that it does not update the logs at the end, and it's only testing stuff, and it can't be interrupted.
func testEpochHip(ss *sim.Sim) {
	ss.Trainer.OnEpochStart()
	for ss.TestEnv.Trial().Cur = 0; ss.TestEnv.Trial().Cur < ss.TestEnv.Trial().Max; ss.TestEnv.Trial().Cur += 1 {
		ss.LoopTrial(etime.Epoch)
	}
	ss.Trainer.OnEpochEnd()
}

// TestAllConditions does a single epoch of testing.
func TestAllConditions(ss *sim.Sim) {
	ss.Trainer.EvalMode = etime.Test
	ss.Trainer.CurEnv = &ss.TestEnv
	ss.Logs.ResetLog(etime.Test, etime.Trial)

	ss.TestEnv.AssignTable(string(TestAB))
	ss.TestEnv.Init(ss.Run.Cur)
	testEpochHip(ss)

	ss.TestEnv.AssignTable(string(TestAC))
	ss.TestEnv.Init(ss.Run.Cur)
	testEpochHip(ss)

	// TODO Commented out because it's slow and unnecessary.
	//ss.TestEnv.AssignTable(string(TestLure))
	//ss.TestEnv.Init(ss.Run.Cur)
	//testEpochHip(ss)

	// Log only after all conditions have been tested.
	ss.Log(ss.Trainer.EvalMode, etime.Epoch)
}

func AddHipCallbacks(ss *HipSim) {
	// Testing
	ss.Trainer.Callbacks = append(ss.Trainer.Callbacks, sim.TrainingCallbacks{
		OnEpochEnd: func() {
			if ss.Trainer.EvalMode == etime.Train {
				if (ss.TestInterval > 0) && ((ss.TrainEnv.Epoch().Cur+1)%ss.TestInterval == 0) {
					// Unlike some testing setups, we only do one epoch of test instead of a whole run.
					TestAllConditions(&ss.Sim)
					ss.Trainer.EvalMode = etime.Train // Important to set these back because TestEpoch sets them to Test.
					ss.Trainer.CurEnv = &ss.TrainEnv
				}
			}
		},
	})

	// These are assigned to actual values at the top of the theta cycle, below. Since we are using closure
	var ca1 *axon.Layer
	var ca3 *axon.Layer
	//var ecin *axon.Layer
	var ecout *axon.Layer
	var ca1FmECin *axon.Prjn
	var ca1FmCa3 *axon.Prjn
	var ca3FmDg *axon.Prjn

	absGain := float32(2)

	// Notes on durations: / 100, 25, 25, 50 best so far, vs 75,50 at start, 50,50 instead of 25..
	//	// cycPerQtr := []int{100, 1, 1, 50} // 150, 1, 1, 50 works for EcCa1Prjn, but 100, 1, 1, 50 does not

	var dgwtscale float32

	// Override Default Phases
	ss.Trainer.Phases = []sim.ThetaPhase{sim.ThetaPhase{
		Name:     "Q1",
		Duration: 50,
		PhaseEnd: func() {
			// Second, Third Quarters: CA1 is driven by CA3 recall
			ss.Net.ActSt1(&ss.Time)
			ca1FmECin.PrjnScale.Abs = 0
			ca1FmCa3.PrjnScale.Abs = absGain
			if ss.Trainer.EvalMode == etime.Train {
				ca3FmDg.PrjnScale.Rel = dgwtscale // restore after 1st quarter
			} else {
				ca3FmDg.PrjnScale.Rel = dgwtscale - ss.Hip.MossyDelTest //TODO 3 Should be replaced with HipSim.MossyDel, and that brings up doubts about our overall approach to HipSim
				//ca3FmDg.PrjnScale.Rel = dgwtscale - ss.Hip.MossyDelTest // testing
			}
			ss.Net.InitGScale() // update computed scaling factors
		},
	}, {
		Name:     "Q2",
		Duration: 50,
		PhaseEnd: func() {
			ss.Net.ActSt2(&ss.Time)
		},
	}, {
		Name:     "Q3",
		Duration: 50,
		PhaseEnd: func() { // Fourth Quarter: CA1 back to ECin drive only
			train := ss.Trainer.EvalMode == etime.Train
			if train { // clamp ECout from ECin
				ca1FmECin.PrjnScale.Abs = absGain
				ca1FmCa3.PrjnScale.Abs = 0
				ss.Net.InitGScale() // update computed scaling factors
				// ecin.UnitVals(&ss.TmpVals, "Act")
				// ecout.ApplyExt1D32(ss.TmpVals)
			}
			ss.Net.MinusPhase(&ss.Time)

			ss.MemStats(train) // must come after QuarterFinal
		},
	}, sim.ThetaPhase{
		Name:     "Q4",
		Duration: 50,
		PhaseEnd: func() {
			ss.Net.PlusPhase(&ss.Time)
		},
	}}

	// Hip Theta Cycle
	ss.Trainer.Callbacks = append(ss.Trainer.Callbacks, sim.TrainingCallbacks{
		OnThetaStart: func() {
			// These are captured at the top of theta cycle, and then referred to in quarters. If you assign them outside these callbacks, then you risk the network being reinitialized, and these pointing to old versions.
			ca1 = ss.Net.LayerByName("CA1").(axon.AxonLayer).AsAxon()
			ca3 = ss.Net.LayerByName("CA3").(axon.AxonLayer).AsAxon()
			// ecin := ss.Net.LayerByName("ECin").(axon.AxonLayer).AsAxon()
			ecout = ss.Net.LayerByName("ECout").(axon.AxonLayer).AsAxon()
			ca1FmECin = ca1.RcvPrjns.SendName("ECin").(axon.AxonPrjn).AsAxon()
			ca1FmCa3 = ca1.RcvPrjns.SendName("CA3").(axon.AxonPrjn).AsAxon()
			ca3FmDg = ca3.RcvPrjns.SendName("DG").(axon.AxonPrjn).AsAxon()

			// First Quarter: CA1 is driven by ECin, not by CA3 recall
			// (which is not really active yet anyway)
			ca1FmECin.PrjnScale.Abs = absGain
			ca1FmCa3.PrjnScale.Abs = 0

			dgwtscale = ca3FmDg.PrjnScale.Rel

			//ca3FmDg.PrjnScale.Rel = dgwtscale - ss.Hip.MossyDel
			ca3FmDg.PrjnScale.Rel = dgwtscale - 3 // turn off DG input to CA3 in first quarter // TODO 3 Should be replaced with HipSim.MossyDel, and that brings up doubts about our overall approach to HipSim

			if ss.Trainer.EvalMode == etime.Train {
				ecout.SetType(emer.Target) // clamp a plus phase during testing todo: ask randy why this is the case
			} else {
				ecout.SetType(emer.Compare) // don't clamp
			}
			ecout.UpdateExtFlags() // call this after updating type

			ss.Net.InitGScale() // update computed scaling factors

		},
		OnThetaEnd: func() {
			ca3FmDg.PrjnScale.Rel = dgwtscale // restore
			ca1FmCa3.PrjnScale.Abs = absGain
		},
		OnMillisecondEnd: func() {
			if ss.Trainer.EvalMode != etime.Train {
				ss.Log(etime.Test, etime.Cycle)
			}
		},
		OnEveryPhaseEnd: func() {
			if ss.GetViewUpdate() == etime.Phase {
				ss.GUI.UpdateNetView()
			}
		},
	})
}

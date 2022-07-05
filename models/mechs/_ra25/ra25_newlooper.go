//Copyright (c) 2019, The Emergent Authors. All rights reserved.
//Use of this source code is governed by a BSD-style
//license that can be found in the LICENSE file.

//go:build notused

package main

import (
	"fmt"
	"log"

	"github.com/Astera-org/obelisk/models/library/common"
	"github.com/Astera-org/obelisk/models/library/sim"
	"github.com/emer/axon/axon"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/envlp"
	_ "github.com/emer/emergent/envlp"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/looper"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/patgen"
	"github.com/emer/emergent/prjn"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
)

var TestEnv = EnvRa25{}
var TrainEnv = EnvRa25{}

var programName = "RA25"
var sizeOfGrid = 6 // By default, 5 by 5 is 25
var numOn = 6      // Number of bits set to 1 in each pattern
var numInputs = 30

// TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
// accum is true.  Note that we're accumulating stats here on the Sim side so the
// core algorithm side remains as simple as possible, and doesn't need to worry about
// different time-scales over which stats could be accumulated etc.
// You can also aggregate directly from log data, as is done for testing stats
func TrialStats(ss *sim.Sim, accum bool) {

	out := ss.Net.LayerByName("Output").(axon.AxonLayer).AsAxon()

	ss.Stats.SetFloat("TrlCosDiff", float64(out.CosDiff.Cos))
	ss.Stats.SetFloat("TrlUnitErr", out.PctUnitErr())

	if ss.Stats.Float("TrlUnitErr") > 0 {
		ss.Stats.SetFloat("TrlErr", 1)
	} else {
		ss.Stats.SetFloat("TrlErr", 0)
	}
}

func main() {
	// TheSim is the overall state for this simulation
	var TheSim sim.Sim
	TheSim.New()

	Config(&TheSim)

	if TheSim.CmdArgs.NoGui {
		TheSim.RunFromArgs() // simple assumption is that any args = no gui -- could add explicit arg if you want
	} else {
		gimain.Main(func() { // this starts gui -- requires valid OpenGL display connection (e.g., X11)
			window := TheSim.ConfigGui(programName, "RA25", "random mapping")
			sim.GuiRun(&TheSim, window)
		})
	}
}

// Config configures all the elements using the standard functions
func Config(ss *sim.Sim) {
	ConfigPats(ss)
	//OpenPats(ss)
	ConfigParams(ss)
	// Parse arguments before configuring the network and env, in case parameters are set.
	ss.ParseArgs()
	ConfigEnv(ss)
	ConfigNet(ss, ss.Net)
	ss.InitStats()
	ss.ConfigLogItems()
	ss.ConfigLogs()
	common.AddDefaultTrainCallbacks(ss)
	common.AddSimpleCallbacks(ss)

}

// ConfigParams configure the parameters
func ConfigParams(ss *sim.Sim) {
	ss.Params.AddNetwork(ss.Net)
	ss.Params.AddSim(ss)
	ss.Params.AddNetSize()

	// ParamSetsMin sets the minimal non-default params
	// Base is always applied, and others can be optionally selected to apply on top of that
	ss.Params.Params = params.Sets{
		{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
			"NetSize": &params.Sheet{
				{Sel: ".Hidden", Desc: "all hidden layers",
					Params: params.Params{
						"Layer.X": "10", //todo layer size correspondence between areas that are connected upstream parameter - get there when we get there
						"Layer.Y": "10",
					},
					Hypers: params.Hypers{
						//"Layer.X": {"StdDev": "0.3", "Min": "2", "Type": "Int"},
						//"Layer.Y": {"StdDev": "0.3", "Min": "2", "Type": "Int"},
					},
				},
			},
			"Network": &params.Sheet{
				{Sel: "Layer", Desc: "all defaults",
					Params: params.Params{
						// All params with importance >=5 have hypers
						"Layer.Inhib.Layer.Gi": "1.2", // 1.2 > 1.1     importance: 10
						// TODO This param should vary with Gi it looks like
						"Layer.Inhib.ActAvg.Init": "0.04", // 0.04 for 1.2, 0.08 for 1.1  importance: 10
						"Layer.Inhib.Layer.Bg":    "0.3",  // 0.3 > 0.0   importance: 2
						"Layer.Act.Decay.Glong":   "0.6",  // 0.6   importance: 2
						"Layer.Act.Dend.GbarExp":  "0.2",  // 0.2 > 0.1 > 0   importance: 5
						"Layer.Act.Dend.GbarR":    "3",    // 3 > 2 good for 0.2 -- too low rel to ExpGbar causes fast ini learning, but then unravels importance: 5
						"Layer.Act.Dt.VmDendTau":  "5",    // 5 > 2.81 here but small effect importance: 1
						"Layer.Act.Dt.VmSteps":    "2",    // 2 > 3 -- somehow works better importance: 1
						"Layer.Act.Dt.GeTau":      "5",    // importance: 1
						"Layer.Act.NMDA.Gbar":     "0.16", //  importance: 7
						"Layer.Act.NMDA.MgC":      "1.4",
						"Layer.Act.NMDA.Voff":     "5",
						"Layer.Act.GABAB.Gbar":    "0.18", // 0.2 > 0.15  importance: 7
					}, Hypers: params.Hypers{
						// These shouldn't be set without also searching for the same value in specific layers like #Input, because it'll clobber them, since it's in a separate Params sheet.
						//"Layer.Inhib.Layer.Gi":    {"StdDev": "0.15"},
						//"Layer.Inhib.ActAvg.Init": {"StdDev": "0.02", "Min": "0.01"},

						//"Layer.Act.Dend.GbarExp":  {"StdDev": "0.05"},
						//"Layer.Act.Dend.GbarR":    {"StdDev": "1"},
						"Layer.Act.NMDA.Gbar":  {"StdDev": "0.05"},
						"Layer.Act.GABAB.Gbar": {"StdDev": "0.05"},
					}},
				{Sel: "#Input", Desc: "critical now to specify the activity level",
					Params: params.Params{
						"Layer.Inhib.Layer.Gi":    "0.94", // 0.9 > 1.0
						"Layer.Act.Clamp.Ge":      "1.08", // 1.0 > 0.6 >= 0.7 == 0.5
						"Layer.Inhib.ActAvg.Init": "0.15", // .24 nominal, lower to give higher excitation
					},
					Hypers: params.Hypers{
						"Layer.Inhib.Layer.Gi": {"StdDev": ".1", "Min": "0", "Priority": "2", "Scale": "LogLinear"},
						"Layer.Act.Clamp.Ge":   {"StdDev": ".2"},
					}},
				{Sel: "#Output", Desc: "output definitely needs lower inhib -- true for smaller layers in general",
					Params: params.Params{
						"Layer.Inhib.Layer.Gi":    "0.9",  // 0.9 >= 0.8 > 1.0 > 0.7 even with adapt -- not beneficial to start low
						"Layer.Inhib.ActAvg.Init": "0.24", // this has to be exact for adapt
						"Layer.Act.Spike.Tr":      "1",    // 1 is new minimum.
						"Layer.Act.Clamp.Ge":      "0.6",  // .6 > .5 v94
						// "Layer.Act.NMDA.Gbar":     "0.3",  // higher not better
					}},
				{Sel: "Prjn", Desc: "norm and momentum on works better, but wt bal is not better for smaller nets",
					Params: params.Params{
						"Prjn.Learn.Lrate.Base":    "0.08", // 0.04 no rlr, 0.2 rlr; .3, WtSig.Gain = 1 is pretty close  //importance: 10
						"Prjn.SWt.Adapt.Lrate":     "0.08", // .1 >= .2, but .2 is fast enough for DreamVar .01..  .1 = more minconstraint //importance: 5
						"Prjn.SWt.Init.SPct":       "0.65", // .5 >= 1 here -- 0.5 more reliable, 1.0 faster..  //importance: 7
						"Prjn.Learn.KinaseCa.Rule": "SynSpkTheta",
					},
					Hypers: params.Hypers{
						"Prjn.Learn.Lrate.Base": {"StdDev": "0.1"},
						//"Prjn.SWt.Adapt.Lrate":  {"StdDev": "0.025"},
						"Prjn.SWt.Init.SPct": {"StdDev": "0.25", "Min": "0.1"},
					}},
				{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
					Params: params.Params{
						"Prjn.PrjnScale.Rel": "0.3", // 0.3 > 0.2 > 0.1 > 0.5 //importance: 9
					},
					Hypers: params.Hypers{
						"Prjn.PrjnScale.Rel": {"StdDev": ".2", "Min": "0.01"},
					}},
			},
			"Sim": &params.Sheet{ // sim params apply to sim object
				{Sel: "Sim", Desc: "best params always finish in this time",
					Params: params.Params{
						"Sim.CmdArgs.MaxEpcs": "100",
					}},
			},
		}},
	}
}

// ApplyInputs applies input patterns from given envirbonment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
// This function should maybe be moved to Environment.
func ApplyInputs(ss *sim.Sim, env envlp.Envs) {
	// TODO This was not being done in RA25; is it ok to do?
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway
	lays := []string{"Input", "Output"}
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		pats := ss.Env().State(ly.Nm)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////
//// 		Configs

func ConfigEnv(ss *sim.Sim) {

	ss.TestEnv = &TestEnv
	ss.TrainEnv = &TrainEnv

	ss.TrialStatsFunc = TrialStats

	ss.NZeroStop = 5

	TrainEnv.Nm = "TrainEnv"
	TrainEnv.Dsc = "training params and state"
	TrainEnv.Table = etable.NewIdxView(ss.Pats)
	ss.TrainEnv.Validate()
	ss.Run.Max = ss.CmdArgs.MaxRuns // note: we are not setting epoch max -- do that manually
	TrainEnv.Epoch().Max = ss.CmdArgs.MaxEpcs

	TestEnv.Nm = "TestEnv"
	TestEnv.Dsc = "testing params and state"
	TestEnv.Table = etable.NewIdxView(ss.Pats)
	TestEnv.SetSequential(true)
	ss.TestEnv.Validate()
	TestEnv.Epoch().Max = ss.CmdArgs.MaxEpcs
	TestEnv.Run().Max = ss.CmdArgs.MaxRuns

	// note: to create a train / test split of pats, do this:
	// all := etable.NewIdxView(ss.Pats)
	// splits, _ := split.Permuted(all, []float64{.8, .2}, []string{"Train", "Test"})
	// ss.TrainEnv.Table = splits.Splits[0]
	// ss.TestEnv.Table = splits.Splits[1]

	ss.TrainEnv.Init(0)
	ss.TestEnv.Init(0)
}

//ConfigPats used to configure patterns
func ConfigPats(ss *sim.Sim) {
	dt := ss.Pats
	dt.SetMetaData("name", "TrainPats")
	dt.SetMetaData("desc", "Training patterns")
	// TODO Make 5 a variable up at the top.
	sch := etable.Schema{
		{"Name", etensor.STRING, nil, nil},
		{"Input", etensor.FLOAT32, []int{sizeOfGrid, sizeOfGrid}, []string{"Y", "X"}},
		{"Output", etensor.FLOAT32, []int{sizeOfGrid, sizeOfGrid}, []string{"Y", "X"}},
	}
	dt.SetFromSchema(sch, numInputs)

	patgen.PermutedBinaryRows(dt.Cols[1], numOn, 1, 0)
	patgen.PermutedBinaryRows(dt.Cols[2], numOn, 1, 0)
	dt.SaveCSV("random_5x5_25_gen.tsv", etable.Tab, etable.Headers)
}

func OpenPats(ss *sim.Sim) {
	dt := ss.Pats
	dt.SetMetaData("name", "TrainPats")
	dt.SetMetaData("desc", "Training patterns")
	err := dt.OpenCSV("random_5x5_25.tsv", etable.Tab)
	if err != nil {
		log.Println(err)
	}
}

func ConfigNet(ss *sim.Sim, net *axon.Network) {
	ss.Params.AddLayers([]string{"Hidden1", "Hidden2"}, "Hidden")
	ss.Params.SetObject("NetSize")

	net.InitName(net, programName) // TODO this should have a name that corresponds to project, leaving for now as it will cause a problem in optimize
	inp := net.AddLayer2D("Input", sizeOfGrid, sizeOfGrid, emer.Input)
	hid1 := net.AddLayer2D("Hidden1", ss.Params.LayY("Hidden1", 10), ss.Params.LayX("Hidden1", 10), emer.Hidden)
	hid2 := net.AddLayer2D("Hidden2", ss.Params.LayY("Hidden2", 10), ss.Params.LayX("Hidden2", 10), emer.Hidden)
	out := net.AddLayer2D("Output", sizeOfGrid, sizeOfGrid, emer.Target)

	// use this to position layers relative to each other
	// default is Above, YAlign = Front, XAlign = Center
	// hid2.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Hidden1", YAlign: relpos.Front, Space: 2})

	// note: see emergent/prjn module for all the options on how to connect
	// NewFull returns a new prjn.Full connectivity pattern
	full := prjn.NewFull()

	net.ConnectLayers(inp, hid1, full, emer.Forward)
	net.BidirConnectLayers(hid1, hid2, full)
	net.BidirConnectLayers(hid2, out, full)

	// net.LateralConnectLayerPrjn(hid1, full, &axon.HebbPrjn{}).SetType(emer.Inhib)

	// note: can set these to do parallel threaded computation across multiple cpus
	// not worth it for this small of a model, but definitely helps for larger ones
	// if Thread {
	// 	hid2.SetThread(1)
	// 	out.SetThread(1)
	// }

	// note: if you wanted to change a layer type from e.g., Target to Compare, do this:
	// out.SetType(emer.Compare)
	// that would mean that the output layer doesn't reflect target values in plus phase
	// and thus removes error-driven learning -- but stats are still computed.

	net.Defaults()
	ss.Params.SetObject("Network")
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	net.InitWts()
}

// ConfigLoops configures the control loops
func ConfigLoops(ss *sim.Sim) {
	//MakeBasicLooper ()
	//MakeBasicLooper(&ss.TrainEnv)
	//ss.

	trn := looper.NewStackEnv(&ss.TrainEnv2) //Handle TrainEnvironment iteration -
	tst := looper.NewStackEnv(&ss.TestEnv2)  // Handle testEnvrionment iteration
	ss.Loops.AddStack(trn)
	ss.Loops.AddStack(tst)

	axon.ConfigLoopsStd(&ss.Loops, ss.Net, &ss.Time, 150, 50) //
	// note: AddCycle0 adds in reverse order of where things end up!
	axon.AddCycle0(&ss.Loops, &ss.Time, "Sim:ApplyInputs", ApplyInputs)
	axon.AddCycle0(&ss.Loops, &ss.Time, "Sim:NewRun", func() {
		if ss.NeedsNewRun {
			ss.NewRun()
		}
	})
	// note: AddLoopCycle adds in reverse order of where things end up!
	if !ss.NoGui { // todo: cmdline
		axon.AddLoopCycle(&ss.Loops, "GUI:UpdateNetView", ss.UpdateNetViewCycle)
		axon.AddLoopCycle(&ss.Loops, "GUI:RasterRec", ss.RasterRec)
	}
	tst.Loop(etime.Cycle).Main.InsertAfter("Axon:Cycle:Run", "Log:Test:Cycle", func() {
		ss.Log(etime.Test, etime.Cycle)
	})
	axon.AddLoopCycle(&ss.Loops, "Sim:SaveState", func() {
		if ss.Time.Phase == 0 {
			switch ss.Time.Cycle { // save states at beta-frequency -- not used computationally
			case 75:
				ss.Net.ActSt1(&ss.Time)
			case 100:
				ss.Net.ActSt2(&ss.Time)
			}
		}
	})
	axon.AddLoopCycle(&ss.Loops, "Sim:StatCounters", ss.StatCounters) // add last so comes first!

	axon.AddPhaseMain(&ss.Loops, "Sim:TrialStats", func() {
		if ss.Time.Phase == 1 {
			ss.TrialStats()
		}
	})

	if !ss.NoGui {
		// after dwt updated, grab it
		trn.Loop(etime.Phase).End.Add("GUI:UpdateNetView", ss.UpdateNetViewCycle)
		tst.Loop(etime.Phase).End.Add("GUI:UpdatePlot", func() {
			ss.GUI.UpdatePlot(etime.Test, etime.Cycle) // make sure always updated at end
		})
	}

	// prepend = before counter is incremented
	trn.Loop(etime.Trial).Main.Prepend("Log:Train:Trial", func() {
		ss.Log(etime.Train, etime.Trial)
	})
	trn.Loop(etime.Epoch).Main.Prepend("Log:Train:Epoch", func() {
		epc := ss.TrainEnv.Counter(etime.Epoch).Cur
		if (ss.TestInterval > 0) && (epc%ss.TestInterval == 0) { // note: epc is *next* so won't trigger first time
			ss.TestAll()
		}
		ss.Log(etime.Train, etime.Epoch)
	})

	trn.Loop(etime.Epoch).Stop.Add("Epoch:NZeroStop", func() bool { // early stopping
		return ss.NZeroStop > 0 && ss.Stats.Int("NZero") >= ss.NZeroStop
	})

	trn.Loop(etime.Run).Main.Prepend("Log:Train:Run", func() {
		ss.Log(etime.Train, etime.Run)
		if ss.SaveWts {
			fnm := ss.WeightsFileName()
			fmt.Printf("Saving Weights to: %s\n", fnm)
			ss.Net.SaveWtsJSON(gi.FileName(fnm))
		}
		ss.NeedsNewRun = true // next step will trigger new init
	})

	tst.Loop(etime.Trial).Main.Add("Log:Test:Trial", func() {
		ss.Log(etime.Test, etime.Trial)
		ss.GUI.NetDataRecord()
	})
	tst.Loop(etime.Epoch).Main.Add("Log:Test:Epoch", func() {
		ss.Log(etime.Test, etime.Epoch)
	})

	if !ss.NoGui {
		trn.Loop(etime.Trial).Main.Prepend("GUI:UpdateNetView", func() {
			ss.UpdateNetViewTime(etime.Trial)
		})
		trn.Loop(etime.Epoch).Main.Prepend("GUI:UpdateNetView", func() {
			ss.UpdateNetViewTime(etime.Epoch)
		})
		tst.Loop(etime.Trial).Main.Prepend("GUI:UpdateNetView", func() {
			ss.UpdateNetViewTime(etime.Trial)
		})
		tst.Loop(etime.Epoch).Main.Prepend("GUI:UpdateNetView", func() {
			ss.UpdateNetViewTime(etime.Epoch)
		})
	}

	fmt.Println(trn.DocString())
	fmt.Println(tst.DocString())
}

package sim

import (
	"bytes"
	"fmt"
	"github.com/emer/emergent/etime"
	"log"

	"github.com/emer/axon/axon"
	_ "github.com/emer/etable/etable"
	"github.com/goki/gi/gi"
)

////////////////////////////////////////////////////////////////////////////////
// 	    Cycles related to running the Network at different timescales.
// 		File organized from fastest timescales to slowest, train then test.

// ThetaCyc runs one theta cycle (200 msec) of processing.
// External inputs must have already been applied prior to calling,
// using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
// If train is true, then learning DWt or WtFmDWt calls are made.
// Handles netview updating within scope, and calls TrainStats()
func (ss *Sim) ThetaCyc(stopScale etime.Times) {
	train := ss.Trainer.EvalMode == etime.Train

	if ss.Time.Cycle == 0 {
		ss.Net.NewState()
		ss.Time.NewState(etime.Train.String())
		ss.Trainer.OnThetaStart()
	}

	for _, phase := range ss.Trainer.Phases {
		if phase.PhaseStart != nil {
			phase.PhaseStart()
		}
		for ; ss.Time.PhaseCycle < phase.Duration; ss.Time.CycleInc() {

			ss.Net.Cycle(&ss.Time)

			// TODO This block should be in Callbacks
			ss.UpdateNetViewText(train)

			// Configuring Train/Cycle log items might be slow.
			ss.Log(ss.Trainer.EvalMode, etime.Cycle)

			if ss.GUI.Active {
				ss.RasterRec(ss.Time.Cycle)
			}

			ss.Trainer.OnMillisecondEnd()

			if stopScale == etime.Cycle {
				ss.GUI.StopNow = true
				ss.Time.CycleInc()
			}
			if ss.GUI.StopNow == true {
				return
			}

		}
		ss.Time.PhaseCycle = 0
		if phase.PhaseEnd != nil {
			phase.PhaseEnd()
		}

		ss.Trainer.OnEveryPhaseEnd()
	}
	ss.Time.Cycle = 0

	ss.TrialStatsFunc(ss, train)
	ss.UpdateNetViewText(train)

	if !train {
		ss.GUI.UpdatePlot(etime.Test, etime.Cycle) // make sure always updated at end
	}
	ss.Trainer.OnThetaEnd()
}

// ApplyInputs applies input patterns from given envirbonment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
// This function should maybe be moved to Environment.
func (ss *Sim) ApplyInputs(env Environment) {
	// TODO This was not being done in RA25; is it ok to do?
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	lays := env.InputAndOutputLayers()
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		pats := (env).State(ly.Nm)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

func (ss *Sim) LoopTrial(stopScale etime.Times) {
	if (*ss.Trainer.CurEnv).Trial().Cur == -1 {
		// This is a hack, and it should be initialized at 0
		(*ss.Trainer.CurEnv).Trial().Cur = 0
	}

	ss.UpdateNetViewText(true)
	ss.ApplyInputs(*ss.Trainer.CurEnv)

	ss.ThetaCyc(stopScale)

	if ss.GUI.StopNow == true {
		return
	}

	ss.Log(ss.Trainer.EvalMode, etime.Trial)

	ss.Trainer.OnTrialEnd()

	if ss.Trainer.EvalMode == etime.Test {
		if ss.CmdArgs.NetData != nil { // offline record net data from testing, just final state
			ss.CmdArgs.NetData.Record(ss.GUI.NetViewText)
		}
	}
}

// LoopEpoch runs until the end of the Epoch, then updates logs.
func (ss *Sim) LoopEpoch(stopScale etime.Times) {
	if (*ss.Trainer.CurEnv).Trial().Cur == 0 {
		ss.Logs.ResetLog(ss.Trainer.EvalMode, etime.Trial)
		ss.Trainer.OnEpochStart()
	}
	for ; (*ss.Trainer.CurEnv).Trial().Cur < (*ss.Trainer.CurEnv).Trial().Max; (*ss.Trainer.CurEnv).Trial().Cur += 1 {
		ss.LoopTrial(stopScale)
		if stopScale == etime.Trial {
			ss.GUI.StopNow = true
			(*ss.Trainer.CurEnv).Trial().Cur += 1
		}
		if ss.GUI.StopNow == true {
			return
		}
	}
	(*ss.Trainer.CurEnv).Trial().Cur = 0

	ss.Trainer.OnEpochEnd()
	// Log after OnEpochEnd.
	ss.Log(ss.Trainer.EvalMode, etime.Epoch)
}

// NewRun intializes a new run of the model, using the ss.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ss.InitRndSeed()
	TrainEnv := ss.TrainEnv
	TestEnv := ss.TestEnv

	run := ss.Run.Cur
	TrainEnv.Init(run)
	TestEnv.Init(run)
	ss.Time.Reset()
	ss.InitRndSeed() //todo should be removed, for debuggin pruposes
	ss.Net.InitWts()
	ss.LoadPretrainedWts()
	ss.InitStats()
	ss.UpdateNetViewText(true)

	// TODO Should this reset all non-run logs?
	ss.Logs.ResetLog(etime.Train, etime.Epoch)
	ss.Logs.ResetLog(etime.Test, etime.Epoch)
	ss.CmdArgs.NeedsNewRun = false
}

// RunEnd is called at the end of a run -- save weights, record final log, etc here
func (ss *Sim) RunEnd() {
	ss.Log(ss.Trainer.EvalMode, etime.Run)

	ss.Trainer.OnRunEnd()
}

func (ss *Sim) loopRun(stopScale etime.Times) {
	if (*ss.Trainer.CurEnv).Epoch().Cur <= 0 && (*ss.Trainer.CurEnv).Trial().Cur <= 0 && ss.Time.Cycle <= 0 {
		if ss.Trainer.EvalMode == etime.Train {
			ss.NewRun()
		}
	}
	if (*ss.Trainer.CurEnv).Trial().Cur == -1 {
		// This is a hack, and it should be initialized at 0
		(*ss.Trainer.CurEnv).Trial().Cur = 0
	}
	// TODO Put "|| ss.Trainer.RunStopEarly()" in conditional, verify
	for ; (*ss.Trainer.CurEnv).Epoch().Cur < (*ss.Trainer.CurEnv).Epoch().Max; (*ss.Trainer.CurEnv).Epoch().Cur += 1 {
		ss.LoopEpoch(stopScale)
		ss.UpdateNetViewText(true)
		if stopScale == etime.Epoch {
			ss.GUI.StopNow = true
			(*ss.Trainer.CurEnv).Epoch().Cur += 1
		}
		if ss.Trainer.RunStopEarly() {
			// End this run early
			break
		}
		if ss.GUI.StopNow == true {
			return
		}
	}
	(*ss.Trainer.CurEnv).Epoch().Cur = 0
	if ss.Trainer.EvalMode == etime.Train {
		ss.RunEnd()
	}
}

// Train trains until the end of runs, unless stopped early by the GUI. Will stop after the end of one unit of time if indicated by stopScale.
// TODO Create a TimeScales for never stop.
func (ss *Sim) Train(stopScale etime.Times) {
	ss.Trainer.EvalMode = etime.Train
	ss.Trainer.CurEnv = &ss.TrainEnv
	// Note that Run, Epoch, and Trial are not initialized at zero to allow Train to restart where it left off.
	for ; ss.Run.Cur < ss.Run.Max; ss.Run.Cur += 1 {
		ss.loopRun(stopScale) // This might set StopNow to true
		if stopScale == etime.Run {
			ss.GUI.StopNow = true
			ss.Run.Cur += 1
		}
		if ss.GUI.StopNow == true {
			ss.GUI.Stopped()
			//ss.GUI.UpdateNetView() // TODO is this necessary?
			// Reset the Stop flag as we leave training.
			ss.GUI.StopNow = false
			return
		}
	}
	// Run.Cur will remain at Run.Max
	ss.GUI.Stopped()
	ss.GUI.UpdateNetView()
}

// SaveWeights saves the network weights -- when called with giv.CallMethod
// it will auto-prompt for filename
func (ss *Sim) SaveWeights(filename gi.FileName) {
	ss.Net.SaveWtsJSON(filename)
}

////////////////////////////////////////////////////////////////////////////////////////////
// Testing

// TestItem tests given item which is at given index in test item list
func (ss *Sim) TestItem(idx int) {
	ss.Trainer.EvalMode = etime.Test
	ss.Trainer.CurEnv = &ss.TrainEnv
	TestEnv := ss.TestEnv
	cur := TestEnv.Trial().Cur
	TestEnv.Trial().Cur = idx
	ss.ApplyInputs(ss.TestEnv)
	ss.ThetaCyc(etime.TimesN)
	TestEnv.Trial().Cur = cur
}

// TestTrial runs one trial of testing -- always sequentially presented inputs
func (ss *Sim) TestTrial() {
	ss.Trainer.EvalMode = etime.Test
	ss.Trainer.CurEnv = &ss.TestEnv
	// ss.TestEnv.Init(ss.Run.Cur) // TODO Should this happen?
	ss.LoopTrial(etime.TimesN) // Do one trial. No need to advance Epoch or Run.
}

// TestAll runs through the full set of testing items for the current run.
// If you stop testing in the middle, it will restart from the beginning.
// This runs across trials and epochs, but not runs.
func (ss *Sim) TestAll() {
	ss.Trainer.EvalMode = etime.Test
	ss.Trainer.CurEnv = &ss.TestEnv
	ss.TestEnv.Init(ss.Run.Cur)
	ss.loopRun(etime.TimesN) // Do a full run of epochs.
}

// TestEpoch does a single epoch of testing.
func (ss *Sim) TestEpoch() {
	ss.Trainer.EvalMode = etime.Test
	ss.Trainer.CurEnv = &ss.TestEnv
	ss.TestEnv.Init(ss.Run.Cur)
	ss.LoopEpoch(etime.TimesN) // Do one epoch.
}

// RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunTestAll() {
	ss.GUI.StopNow = false
	ss.TestAll()
	ss.GUI.Stopped()
}

func (ss *Sim) LoadPretrainedWts() bool {
	if ss.PreTrainWts == nil {
		return false
	}
	b := bytes.NewReader(ss.PreTrainWts)
	err := ss.Net.ReadWtsJSON(b)
	if err != nil {
		log.Println(err)
	} else {
		fmt.Printf("loaded pretrained wts\n")
	}
	return true
}

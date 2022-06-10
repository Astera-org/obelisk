package common

import (
	"fmt"
	"github.com/Astera-org/models/library/sim"
	"github.com/emer/emergent/etime"
	"github.com/goki/gi/gi"
)

func AddDefaultTrainCallbacks(ss *sim.Sim) {
	// Save Weights
	ss.Trainer.Callbacks = append(ss.Trainer.Callbacks, sim.TrainingCallbacks{
		OnRunEnd: func() {
			if ss.CmdArgs.SaveWts {
				fnm := ss.WeightsFileName()
				fmt.Printf("Saving Weights to: %s\n", fnm)
				ss.Net.SaveWtsJSON(gi.FileName(fnm))
			}
		},
	})

	// Weight storage and update delta.
	ss.Trainer.Callbacks = append(ss.Trainer.Callbacks, sim.TrainingCallbacks{
		Name: "DWt",
		OnThetaStart: func() {
			// update prior weight changes at start, so any DWt values remain visible at end
			// you might want to do this less frequently to achieve a mini-batch update
			// in which case, move it out to the TrainTrial method where the relevant
			// counters are being dealt with.

			// update prior weight changes at start, so any DWt values remain visible at end
			if ss.Trainer.EvalMode == etime.Train {
				// Apply delta weight.
				ss.Net.WtFmDWt(&ss.Time)
			}
		},
		OnThetaEnd: func() {
			if ss.Trainer.EvalMode == etime.Train {
				// Compute delta weight.
				ss.Net.DWt(&ss.Time)
			}
		},
	})

	// Learning Rate Schedule
	ss.Trainer.Callbacks = append(ss.Trainer.Callbacks, sim.TrainingCallbacks{
		OnEpochEnd: func() {
			if ss.Trainer.EvalMode == etime.Train {
				LrateSched(ss, ss.TrainEnv.Epoch().Cur)
			}
		},
	})

	// PCA Stats
	ss.Trainer.Callbacks = append(ss.Trainer.Callbacks, sim.TrainingCallbacks{
		OnEpochEnd: func() {
			if ss.Trainer.EvalMode == etime.Train {
				// Should run on first epoch, needs to run before Log.
				if (ss.PCAInterval > 0) && (ss.TrainEnv.Epoch().Cur%ss.PCAInterval == 0) {
					ss.PCAStats()
				}
			}
		},
		OnTrialEnd: func() {
			if ss.Trainer.EvalMode == etime.Train {
				if (ss.PCAInterval > 0) && (ss.TrainEnv.Epoch().Cur%ss.PCAInterval == 0) {
					ss.Log(etime.Analyze, etime.Trial)
				}
			}
		},
	})

	// First Zero Early Stopping
	ss.Trainer.Callbacks = append(ss.Trainer.Callbacks, sim.TrainingCallbacks{
		RunStopEarly: func() bool {
			return ss.NZeroStop > 0 && ss.Stats.Int("NZero") >= ss.NZeroStop
		},
	})

	AddPlusAndMinusPhases(ss)

	AddDefaultGUICallbacks(ss)
}

func AddDefaultGUICallbacks(ss *sim.Sim) {
	var viewUpdt etime.Times // Reset at the top of theta cycle.
	viewUpdtCallbacks := sim.TrainingCallbacks{
		OnThetaStart: func() {
			// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine

			viewUpdt = ss.TrainUpdt
			if ss.Trainer.EvalMode == etime.Test {
				viewUpdt = ss.TestUpdt
			}
			if viewUpdt == etime.Phase {
				ss.GUI.UpdateNetView()
			}
		},
		OnMillisecondEnd: func() {
			if ss.ViewOn {
				ss.UpdateViewTime(viewUpdt)
			}
		},
	}
	ss.Trainer.Callbacks = append(ss.Trainer.Callbacks, viewUpdtCallbacks)
}

func AddPlusAndMinusPhases(ss *sim.Sim) {
	ss.Trainer.Phases = []sim.ThetaPhase{sim.ThetaPhase{
		Name:     "Minus",
		Duration: 150,
		OnMillisecondEnd: func() {
			switch ss.Time.Cycle { // save states at beta-frequency -- not used computationally
			case 75:
				ss.Net.ActSt1(&ss.Time)
			case 100:
				ss.Net.ActSt2(&ss.Time)
			}
		},
		PhaseStart: func() {
			ss.Time.PlusPhase = false
		},
		PhaseEnd: func() {
			ss.Net.MinusPhase(&ss.Time)
		},
	}, sim.ThetaPhase{
		Name:     "Plus",
		Duration: 50,
		PhaseStart: func() {
			ss.Time.PlusPhase = true
			ss.UpdateNetViewText(ss.Trainer.EvalMode == etime.Train)
		},
		PhaseEnd: func() {
			ss.Net.PlusPhase(&ss.Time)
		},
	}}
}

// LrateSched implements the learning rate schedule
func LrateSched(ss *sim.Sim, epc int) {
	switch epc {
	case 40:
		ss.Net.LrateMod(0.5)
		fmt.Printf("dropped lrate 0.5 at epoch: %d\n", epc)
	}
}

// AddSimpleCallbacks is used by RA25 and one2many.
func AddSimpleCallbacks(ss *sim.Sim) {
	// Testing
	ss.Trainer.Callbacks = append(ss.Trainer.Callbacks, sim.TrainingCallbacks{
		OnEpochEnd: func() {
			if ss.Trainer.EvalMode == etime.Train {
				if (ss.TestInterval > 0) && ((ss.TrainEnv.Epoch().Cur+1)%ss.TestInterval == 0) {
					ss.TestAll()
					ss.Trainer.EvalMode = etime.Train // Important to set these back because TestAll sets them to Test.
					ss.Trainer.CurEnv = &ss.TrainEnv
				}
			}
		},
	})
}

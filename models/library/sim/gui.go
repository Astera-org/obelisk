package sim

import (
	"fmt"
	"github.com/emer/emergent/etime"

	"github.com/emer/emergent/egui"
	"github.com/goki/gi/gi"
	"github.com/goki/ki/ki"
	"github.com/goki/mat32"
)

func GuiRun(TheSim *Sim, window *gi.Window) {
	TheSim.Init()
	//window  //:= TheSim.ConfigGui(appname, title, about)
	window.StartEventLoop()
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui(appname, title, about string) *gi.Window {
	ss.GUI.MakeWindow(ss, appname, title, about)
	ss.GUI.CycleUpdateInterval = 10
	ss.GUI.AddNetView("NetView")
	ss.GUI.NetView.SetNet(ss.Net) // TODO ask Randy what this is doing

	ss.GUI.NetView.Scene().Camera.Pose.Pos.Set(0, 1, 2.75) // more "head on" than default which is more "top down"
	ss.GUI.NetView.Scene().Camera.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})
	ss.GUI.AddPlots(title, &ss.Logs)

	stb := ss.GUI.TabView.AddNewTab(gi.KiT_Layout, "Spike Rasters").(*gi.Layout)
	stb.Lay = gi.LayoutVert
	stb.SetStretchMax()
	for _, lnm := range ss.Stats.Rasters {
		sr := ss.Stats.F32Tensor("Raster_" + lnm)
		ss.GUI.ConfigRasterGrid(stb, lnm, sr)
	}
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Init", Icon: "update",
		Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Init()
			ss.GUI.UpdateWindow()
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Train",
		Icon:    "run",
		Tooltip: "Starts the network training, picking up from wherever it may have left off.  If not stopped, training will complete the specified number of Runs through the full number of Epochs of training, with testing automatically occuring at the specified interval.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				ss.GUI.IsRunning = true
				ss.GUI.ToolBar.UpdateActions()
				go ss.Train(etime.TimesN) // Train until end of all Runs
			}
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Stop",
		Icon:    "stop",
		Tooltip: "Interrupts running.  Hitting Train again will pick back up where it left off.",
		Active:  egui.ActiveRunning,
		Func: func() {
			ss.GUI.StopNow = true
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Step Trial",
		Icon:    "step-fwd",
		Tooltip: "Advances one training trial at a time.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				ss.GUI.IsRunning = true
				ss.Train(etime.Trial)
				ss.UpdateNetViewText(true)
				ss.GUI.IsRunning = false
				ss.GUI.UpdateWindow()
			}
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Step Cycle",
		Icon:    "step-fwd",
		Tooltip: "Advances one cycle at a time.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				ss.GUI.IsRunning = true
				ss.Train(etime.Cycle)
				ss.UpdateNetViewText(true)
				ss.GUI.IsRunning = false
				ss.GUI.UpdateWindow()
			}
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Step Epoch",
		Icon:    "fast-fwd",
		Tooltip: "Advances one epoch (complete set of training patterns) at a time.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				ss.GUI.IsRunning = true
				ss.GUI.ToolBar.UpdateActions()
				ss.GUI.StopNow = false
				go ss.Train(etime.Epoch)
			}
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Step Run",
		Icon:    "fast-fwd",
		Tooltip: "Advances one full training Run at a time.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				ss.GUI.IsRunning = true
				ss.GUI.ToolBar.UpdateActions()
				ss.GUI.StopNow = false
				go ss.Train(etime.Run)
			}
		},
	})

	////////////////////////////////////////////////
	ss.GUI.ToolBar.AddSeparator("test")
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Test Trial",
		Icon:    "fast-fwd",
		Tooltip: "Runs the next testing trial.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				ss.GUI.IsRunning = true
				ss.GUI.StopNow = false
				ss.TestTrial()
				ss.GUI.IsRunning = false
				ss.GUI.UpdateWindow()
			}
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Test Item",
		Icon:    "step-fwd",
		Tooltip: "Prompts for a specific input pattern name to run, and runs it in testing mode.",
		Active:  egui.ActiveStopped,
		Func: func() {

			gi.StringPromptDialog(ss.GUI.ViewPort, "", "Test Item",
				gi.DlgOpts{Title: "Test Item", Prompt: "Enter the Name of a given input pattern to test (case insensitive, contains given string."},
				ss.GUI.Win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
					dlg := send.(*gi.Dialog)
					if sig == int64(gi.DialogAccepted) {
						val := gi.StringPromptDialogValue(dlg)
						idxs := []int{0} //TODO: //ss.TestEnv.Table.RowsByString("Name", val, etable.Contains, etable.IgnoreCase)
						if len(idxs) == 0 {
							gi.PromptDialog(nil, gi.DlgOpts{Title: "Name Not Found", Prompt: "No patterns found containing: " + val}, gi.AddOk, gi.NoCancel, nil, nil)
						} else {
							if !ss.GUI.IsRunning {
								ss.GUI.IsRunning = true
								fmt.Printf("testing index: %d\n", idxs[0])
								ss.TestItem(idxs[0])
								ss.GUI.IsRunning = false
								ss.GUI.ViewPort.SetNeedsFullRender()
							}
						}
					}
				})

		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Test All",
		Icon:    "step-fwd",
		Tooltip: "Prompts for a specific input pattern name to run, and runs it in testing mode.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				ss.GUI.IsRunning = true
				ss.GUI.ToolBar.UpdateActions()
				go ss.RunTestAll()
			}
		},
	})

	////////////////////////////////////////////////
	ss.GUI.ToolBar.AddSeparator("log")
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Reset RunLog",
		Icon:    "reset",
		Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.Logs.ResetLog(etime.Train, etime.Run)
			ss.GUI.UpdatePlot(etime.Train, etime.Run)
		},
	})
	////////////////////////////////////////////////
	ss.GUI.ToolBar.AddSeparator("misc")
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "DefineSimVariables Seed",
		Icon:    "new",
		Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time.",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.NewRndSeed()
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "README",
		Icon:    "file-markdown",
		Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
		Active:  egui.ActiveAlways,
		Func: func() {
			gi.OpenURL("https://github.com/emer/axon/blob/master/examples/one2many/README.md")
		},
	})
	ss.GUI.FinalizeGUI(false)
	return ss.GUI.Win
}

func (ss *Sim) Counters(train bool) string {
	if train {
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%v\t\t\t", ss.Run.Cur, ss.TrainEnv.Epoch().Cur, ss.TrainEnv.Trial().Cur, ss.Time.Cycle, ss.TrainEnv.TrialName().Cur)
	} else {
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%v\t\t\t", ss.Run.Cur, ss.TrainEnv.Epoch().Cur, ss.TestEnv.Trial().Cur, ss.Time.Cycle, ss.TestEnv.TrialName().Cur)
	}
}

// UpdateNetViewText saves a string rep of them to the GUI, if the GUI is active
func (ss *Sim) UpdateNetViewText(train bool) {
	ev := ss.TrainEnv
	if !train {
		ev = ss.TestEnv
	}
	displayText := fmt.Sprintf("%s\tRun:\t%d/%d\tEpoch:%d/%d\tTrial:\t%d/%d\tCycle:\t%d\t", ss.Trainer.EvalMode, ss.Run.Cur, ss.Run.Max, ev.Epoch().Cur, ev.Epoch().Max, ev.Trial().Cur, ev.Trial().Max, ss.Time.Cycle) + "\t" + ss.Stats.Print([]string{"TrlErr", "TrlCosDiff"})
	//println(displayText)
	ss.GUI.NetViewText = displayText
}

func (ss *Sim) UpdateView(train bool) {
	if ss.GUI.NetView != nil && ss.GUI.NetView.IsVisible() {
		ss.GUI.NetView.Record(ss.Counters(train))
		// note: essential to use Go version of update when called from another goroutine
		ss.GUI.NetView.GoUpdate() // note: using counters is significantly slower..
	}
}

// UpdateViewTime based on differetn time scales change the values accoridngly
func (ss *Sim) UpdateViewTime(viewUpdt etime.Times) {
	// If the NetView is flickering and you don't like it, use ss.Time.Cycle+1 here. Network activity is actually reset at zero.
	switch viewUpdt {
	case etime.Cycle:
		ss.GUI.UpdateNetView()
	case etime.FastSpike:
		if ss.Time.Cycle%10 == 0 {
			ss.GUI.UpdateNetView()
		}
	case etime.GammaCycle:
		if ss.Time.Cycle%25 == 0 {
			ss.GUI.UpdateNetView()
		}
	case etime.AlphaCycle:
		if (ss.Time.Cycle+1)%100 == 0 {
			ss.GUI.UpdateNetView()
		}
	}
}

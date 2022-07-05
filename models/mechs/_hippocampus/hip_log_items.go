package main

import (
	"github.com/Astera-org/obelisk/models/library/sim"
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/etime"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/minmax"
	"github.com/emer/etable/split"
)

// InitStats initializes all the statistics.
// called at start of new run
func InitHipStats(ss *sim.Sim) {
	// TODO HIP Copied
	// accumulators
	ss.Stats.SetFloat("SumUnitErr", 0)
	ss.Stats.SetFloat("SumCosDiff", 0)
	ss.Stats.SetInt("CntErr", 0)
	ss.Stats.SetInt("HipFirstZero", -1)
	ss.Stats.SetInt("HipNZero", 0)
	// clear rest just to make Sim look initialized
	ss.Stats.SetFloat("Mem", 0)
	ss.Stats.SetFloat("TrgOnWasOff", 0)
	ss.Stats.SetFloat("TrgOnWasOffCmp", 0)
	ss.Stats.SetFloat("TrgOffWasOn", 0)
	ss.Stats.SetFloat("TrlUnitErr", 0)
	ss.Stats.SetFloat("EpcUnitErr", 0)
	ss.Stats.SetFloat("EpcPctErr", 0)
	ss.Stats.SetFloat("EpcCosDiff", 0)

	// TODO These need to be initialized. Maybe all stats should initialize to 0?
	ss.Stats.SetFloat("MemThr", 0.34)
	ss.Stats.SetInt("NZeroStop", 1)
}

func ConfigHipItems(ss *sim.Sim) {
	// Don't plot other things.
	for _, lg := range ss.Logs.Items {
		lg.Plot = elog.DFalse
	}

	ss.Logs.AddItem(&elog.Item{
		Name:   "Mem",
		Type:   etensor.FLOAT64,
		Plot:   elog.DTrue,
		FixMax: elog.DTrue,
		FixMin: elog.DTrue,
		Range:  minmax.F64{Max: 1},
		Write: elog.WriteMap{
			etime.Scope(etime.AllModes, etime.Trial): func(ctx *elog.Context) {
				ctx.SetStatFloat("Mem")
			},
			etime.Scope(etime.AllModes, etime.Epoch): func(ctx *elog.Context) {
				ctx.SetAgg(ctx.Mode, etime.Trial, agg.AggMean) // TODO how is this referencing Mem name
			},
		}})
	ss.Logs.AddItem(&elog.Item{
		Name:   "TrgOnWasOff",
		Type:   etensor.FLOAT64,
		Plot:   elog.DTrue,
		FixMax: elog.DTrue,
		FixMin: elog.DTrue,
		Range:  minmax.F64{Max: 1},
		Write: elog.WriteMap{
			etime.Scope(etime.AllModes, etime.Trial): func(ctx *elog.Context) {
				ctx.SetStatFloat("TrgOnWasOff")
			},
			etime.Scope(etime.AllModes, etime.Epoch): func(ctx *elog.Context) {
				ctx.SetAgg(ctx.Mode, etime.Trial, agg.AggMean) // TODO how is this referencing Mem name
			},
		}})
	ss.Logs.AddItem(&elog.Item{
		Name:   "TrgOffWasOn",
		Type:   etensor.FLOAT64,
		Plot:   elog.DTrue,
		FixMax: elog.DTrue,
		FixMin: elog.DTrue,
		Range:  minmax.F64{Max: 1},
		Write: elog.WriteMap{
			etime.Scope(etime.AllModes, etime.Trial): func(ctx *elog.Context) {
				ctx.SetStatFloat("TrgOffWasOn")
			},
			etime.Scope(etime.AllModes, etime.Epoch): func(ctx *elog.Context) {
				ctx.SetAgg(ctx.Mode, etime.Trial, agg.AggMean) // TODO how is this referencing Mem name
			},
		}})
	ss.Logs.AddItem(&elog.Item{
		Name: "TestNm",
		Type: etensor.STRING,
		Plot: elog.DFalse,
		Write: elog.WriteMap{
			etime.Scope(etime.Test, etime.Trial): func(ctx *elog.Context) {
				testName := "UNKNOWN"
				// These need to match up with tstNms below.
				if ss.TestEnv.Desc() == "TestAB" {
					testName = "AB"
				}
				if ss.TestEnv.Desc() == "TestAC" {
					testName = "AC"
				}
				if ss.TestEnv.Desc() == "TestLure" {
					testName = "Lure"
				}
				// Handle other cases here.
				ctx.SetString(testName)
			},
		}})

	// Add AB Mem and stuff
	tstNms := []string{"AB", "AC"} // TODO Add in "Lure"
	tstStatNms := []string{"Mem", "TrgOnWasOff", "TrgOffWasOn"}

	for _, tni := range tstNms {
		for _, tsi := range tstStatNms {
			// These holder variables are needed because they are used in a closure below.
			tn := tni
			ts := tsi
			plot := elog.DFalse
			if ts == "Mem" {
				plot = elog.DTrue
			}
			ss.Logs.AddItem(&elog.Item{
				Name:   tn + "_" + ts,
				Type:   etensor.FLOAT64,
				Plot:   plot,
				FixMax: elog.DTrue,
				FixMin: elog.DTrue,
				Range:  minmax.F64{Max: 1},
				Write: elog.WriteMap{
					etime.Scope(etime.Test, etime.Epoch): func(ctx *elog.Context) {
						trl := ss.Logs.Table(etime.Test, etime.Trial)
						trix := etable.NewIdxView(trl)
						spl := split.GroupBy(trix, []string{"TestNm"})
						for _, ts := range tstStatNms {
							split.Agg(spl, ts, agg.AggMean)
						}
						tstStats := spl.AggsToTable(etable.ColNameOnly)

						for ri := 0; ri < tstStats.Rows; ri++ {
							tst := tstStats.CellString("TestNm", ri)
							for _, tso := range tstStatNms {
								if tst+" "+tso == tn+" "+ts {
									ctx.SetFloat64(tstStats.CellFloat(ts, ri))
									break
								}
							}
						}
					},
				}})
		}
	}

	// Run level aggregates
	ss.Logs.AddItem(&elog.Item{
		Name: "ABEndMem",
		Type: etensor.FLOAT64,
		Plot: elog.DTrue,
		Write: elog.WriteMap{
			etime.Scope(etime.Train, etime.Run): func(ctx *elog.Context) {
				ix := ctx.Logs.IdxViewScope(etime.Scope(etime.Test, etime.Epoch))
				lastAbMem := ix.Table.CellFloat("AB_Mem", ix.Table.Rows-1) //.Col(ix.Table.ColIdx("AB_Mem"))
				//for am : abmems.
				// TODO Get AB Mem at last epoch
				ctx.SetFloat64(lastAbMem)
			},
		}})

	ss.Logs.AddItem(&elog.Item{
		// TODO Could be named ACSuccessEpoch, but name is used for SetAgg
		Name: "ACSuccessEpoch",
		Type: etensor.INT64,
		Plot: elog.DTrue,
		Write: elog.WriteMap{
			etime.Scope(etime.Train, etime.Run): func(ctx *elog.Context) {
				// TODO Get num of epoch at stop
				//ctx.SetFloat32(0)
				//ctx.SetAgg(etime.Test, etime.Epoch, agg.AggMax)
				maxo := ctx.SetAggItemScope(etime.Scope(etime.Test, etime.Epoch), "Epoch", agg.AggMax)
				print(maxo) // DO NOT SUBMIT
			},
		}})
}

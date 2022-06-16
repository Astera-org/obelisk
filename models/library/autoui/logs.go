package autoui

import (
	"github.com/emer/axon/axon"
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/etime"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/minmax"
)

// AddCommonLogItemsForOutputLayers helps the UserInterface add common logging items based on the output layers of an axon Network.
func AddCommonLogItemsForOutputLayers(ui *AutoUI) {
	// Record time in logs
	for m, st := range ui.Looper.Stacks {
		mode := m
		for t, l := range st.Loops {
			time := t
			loop := l
			ui.Logs.AddItem(&elog.Item{
				Name: t.String(),
				Type: etensor.INT64,
				Plot: elog.DFalse,
				Write: elog.WriteMap{
					etime.Scopes([]etime.Modes{mode}, []etime.Times{time}): func(ctx *elog.Context) {
						ctx.SetInt(loop.Counter.Cur)
					}}})
		}
	}

	// Error for output layers
	for _, olnm := range ui.Network.LayersByClass(emer.Target.String()) {
		out := ui.Network.LayerByName(olnm).(axon.AxonLayer).AsAxon()

		// TODO These should be computed at the Trial, not Cycle level
		baseComputeLevel := etime.Trial
		found := false
		corSimMap := elog.WriteMap{}
		pctErrMap := elog.WriteMap{}
		trlCorrMap := elog.WriteMap{}
		for m, st := range ui.Looper.Stacks {
			for iter := len(st.Order) - 1; iter >= 0; iter-- {
				i := iter // For closures
				t := st.Order[i]
				if st.Order[iter] == baseComputeLevel {
					found = true // Subsequent layers can do aggregation.
					corSimMap[etime.Scope(m, t)] = func(ctx *elog.Context) {
						ctx.SetFloat32(out.CorSim.Cor)
					}
					pctErrMap[etime.Scope(m, t)] = func(ctx *elog.Context) {
						ctx.SetFloat64(out.PctUnitErr())
					}
					trlCorrMap[etime.Scope(m, t)] = func(ctx *elog.Context) {
						pcterr := out.PctUnitErr()
						trlCorr := 1
						if pcterr > 0 {
							trlCorr = 0
						}
						ctx.SetFloat64(float64(trlCorr))
					}
				} else if found {
					// All other, less frequent, timescales are an aggregate
					for _, wm := range []elog.WriteMap{corSimMap, pctErrMap, trlCorrMap} {
						wm[etime.Scope(m, t)] = func(ctx *elog.Context) {
							ctx.SetAgg(ctx.Mode, st.Order[i+1], agg.AggMean)
						}
					}
				}
			}
		}

		// Add it to the list.
		ui.Logs.AddItem(&elog.Item{
			Name:   olnm + "CorSim",
			Type:   etensor.FLOAT64,
			Plot:   elog.DTrue,
			Range:  minmax.F64{Min: 0, Max: 1},
			FixMax: elog.DTrue,
			Write:  corSimMap})
		ui.Logs.AddItem(&elog.Item{
			Name:  olnm + "PctErr",
			Type:  etensor.FLOAT64,
			Plot:  elog.DTrue,
			Write: pctErrMap})
		ui.Logs.AddItem(&elog.Item{
			Name:  olnm + "UnitCorr",
			Type:  etensor.FLOAT64,
			Plot:  elog.DTrue,
			Write: trlCorrMap})
	}
}

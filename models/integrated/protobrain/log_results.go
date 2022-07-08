package main

import (
	log "github.com/Astera-org/easylog"
	"github.com/Astera-org/obelisk/models/library/autoui"
	"github.com/Astera-org/obelisk/models/library/helpers"
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/etime"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/minmax"
)

// logExtra handles default logging plus the we get back from Fworld
func (ss *Sim) AddExtraLogItemsFWorlds(ui *autoui.AutoUI) {
	autoui.AddCommonLogItemsForOutputLayers(ui)
	f1Map := elog.WriteMap{}
	f1Map[etime.Scope(etime.Train, etime.Trial)] = func(ctx *elog.Context) {
		ctx.SetFloat64(float64(ss.F1Score))
	}
	f1Map[etime.Scope(etime.Train, etime.Epoch)] = func(ctx *elog.Context) {
		ctx.SetFloat64(float64(ss.F1Score))
	}
	// Add it to the list.
	ui.Logs.AddItem(&elog.Item{
		Name:   "F1Culmulative",
		Type:   etensor.FLOAT64,
		Plot:   elog.DTrue,
		Range:  minmax.F64{Min: 0, Max: 1},
		FixMax: elog.DTrue,
		Write:  f1Map})
}

func createActionHistoryRow(predicted, groundtruth, run float64, timescale string) etable.Table {
	table := etable.Table{}
	table.AddCol(helpers.WrapperTensorFloat(predicted), "Predicted")
	table.AddCol(helpers.WrapperTensorFloat(groundtruth), "GroundTruth")
	table.AddCol(helpers.WrapperTensorString(timescale), "Timescale")
	table.AddCol(helpers.WrapperTensorFloat(run), "Run")
	table.SetNumRows(1)
	return table

}

func (ss *Sim) AddActionHistory(observations map[string]etensor.Tensor, timeScale string) {
	bestAction, actionExists := observations["Heuristic"]
	prevPredictedAction, predictedExists := observations["PredictedActionLastTimeStep"]
	if (actionExists == false) || (predictedExists == false) { //if not getting info across network, perhaps using diff world
		log.Error("Heuristic or Predicted Action keys are not found in observations, cannot log results")
	} else {
		floatAction := float64(bestAction.FloatVal1D(0))
		floatPredictedPrevious := float64(prevPredictedAction.FloatVal1D(0))
		runNum := float64(ss.Loops.GetLoop(etime.Train, etime.Run).Counter.Cur)
		dframe := createActionHistoryRow(-1.0, floatAction, runNum, timeScale)
		if ss.ActionHistory.Rows == 0 {
			ss.ActionHistory = &dframe
		} else {
			ss.ActionHistory.AppendRows(&dframe)
			ss.ActionHistory.SetCellFloat("Predicted", ss.ActionHistory.Rows-2, floatPredictedPrevious) //set n-1 one values
		}
	}
}

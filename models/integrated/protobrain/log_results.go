package main

import (
	log "github.com/Astera-org/easylog"
	"github.com/Astera-org/obelisk/models/library/autoui"
	"github.com/Astera-org/obelisk/models/library/helpers"
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/etime"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/minmax"
)

// AddExtraFWorldItems to capture range of values and behavior as well as internal values
func (ss *Sim) AddExtraFWorldItems(ui *autoui.AutoUI) {
	autoui.AddCommonLogItemsForOutputLayers(ui)
	names := []string{"KL", "F1Resources", "Energy", "Hydra", "EatF1", "DrinkF1"}
	for _, name := range names {
		currentName := name
		nameMap := elog.WriteMap{}
		nameMap[etime.Scope(etime.Train, etime.Trial)] = func(ctx *elog.Context) {

			vector := ss.WorldEnv.Observe(currentName) //a vector storing a SINGLE value
			if vector != nil {
				value := vector.FloatVal1D(0)
				ctx.SetFloat64(float64(value))
			} else {
				log.Warn("trying to find a value that doesn't exist in fworld", currentName)
			}
		}
		nameMap[etime.Scope(etime.Train, etime.Epoch)] = func(ctx *elog.Context) {
			ctx.SetAgg(etime.Train, etime.Trial, agg.AggMean)
		}

		ui.Logs.AddItem(&elog.Item{
			Name:   name,
			Type:   etensor.FLOAT64,
			Plot:   elog.DTrue,
			Range:  minmax.F64{Min: 0, Max: 1},
			FixMax: elog.DTrue,
			Write:  nameMap})

	}
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

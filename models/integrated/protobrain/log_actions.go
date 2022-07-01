package main

import (
	log "github.com/Astera-org/easylog"
	"github.com/Astera-org/models/library/helpers"
	"github.com/emer/emergent/etime"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
)

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

package main

import (
	log "github.com/Astera-org/easylog"
	"github.com/Astera-org/models/library/metrics"
	"github.com/emer/emergent/etime"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/goki/gi/gi"
)

func wrapperTensorFloat(value float64) etensor.Tensor {
	predictedTensor := etensor.New(etensor.FLOAT64, []int{1.0}, nil, nil)
	predictedTensor.SetFloat1D(0, float64(value))
	return predictedTensor
}
func wrapperTensorString(value string) etensor.Tensor {
	predictedTensor := etensor.New(etensor.STRING, []int{1}, nil, nil)
	predictedTensor.SetString1D(0, value)

	return predictedTensor
}

func createActionHistoryRow(predicted, groundtruth, run float64, timescale string) etable.Table {
	table := etable.Table{}
	table.AddCol(wrapperTensorFloat(predicted), "Predicted")
	table.AddCol(wrapperTensorFloat(groundtruth), "GroundTruth")
	table.AddCol(wrapperTensorString(timescale), "Timescale")
	table.AddCol(wrapperTensorFloat(run), "Run")
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

// TensorToAryFloat converts etensors to 1d float64 matrices
func TensorToAryFloat(tensor *etensor.Tensor) []float64 {
	ary := make([]float64, (*tensor).Len())
	i := 0
	for i = 0; i < (*tensor).Len(); i++ {
		ary[i] = (*tensor).FloatVal1D(i)
	}
	return ary
}

// TensorToAryFloat converts etensors to 1d float64 matrices
func TensorToAryInt(tensor *etensor.Tensor) []int32 {
	floatary := TensorToAryFloat(tensor)
	ary := make([]int32, len(floatary))
	for i := 0; i < len(floatary); i++ {
		ary[i] = int32(floatary[i])
	}
	return ary
}

// StoreTensors stores a map of tensors in a table for ease of saving and loading back
func StoreTensors(observations map[string]etensor.Tensor) *etable.Table {
	temp := etable.Table{}
	temp.Rows = 1
	for name, value := range observations {
		temp.AddCol(value, name)
	}
	return &temp
}

// WriteActionHistory stores a history of action patterns, removes the last most pattern (since conversion is happening in fworld, and last action won't have a corresponding class)
func WriteActionHistory(sourceTable *etable.Table, filename gi.FileName) {
	if sourceTable.Rows == 0 {
		log.Warn("ActionHistory is empty, check if discrete actions are coming from Fworld")
	} else {
		sourceTable.SetNumRows(sourceTable.NumRows() - 1) //skip last one, since no action correspondance
		sourceTable.SaveCSV(filename, ',', true)
	}
}

// ActionF1Score calculates the F1 score for a given action and expected action, the last most prediction is not logged, so skip it
func ActionF1Score(sourceTable *etable.Table) float64 {

	if sourceTable.Rows == 0 {
		log.Warn("ActionHistory is empty, check if discrete actions are coming from Fworld")
		return -1.0
	} else {
		predictTensor := (sourceTable.ColByName("Predicted"))
		groundTensor := sourceTable.ColByName("GroundTruth")
		predicted := TensorToAryInt(&predictTensor)
		groundtruth := TensorToAryInt(&groundTensor)
		return metrics.F1ScoreMacro(predicted[:len(predicted)-1], groundtruth[:len(groundtruth)-1], []int32{0, 1, 2, 3, 4}) //skip last one, since no action correspondance
	}
}

//	ActionKL calculates kldivergence for action tbale but explicitely skips last most trial where the predicted action may not have been assigned
func ActionKL(sourceTable *etable.Table) float64 {
	if sourceTable.Rows == 0 {
		log.Warn("ActionHistory is empty, check if discrete actions are coming from Fworld")
		return -1.0
	} else {
		predictTensor := (sourceTable.ColByName("Predicted"))
		groundTensor := sourceTable.ColByName("GroundTruth")
		predicted := TensorToAryInt(&predictTensor)
		groundtruth := TensorToAryInt(&groundTensor)
		return metrics.KLDivergence(predicted[:len(predicted)-1], groundtruth[:len(groundtruth)-1])
	}
}

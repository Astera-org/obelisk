package main

import (
	log "github.com/Astera-org/easylog"
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

// TensorToDense converts etensors to 1d float64 matrices
func TensorToDense(tensor etensor.Tensor) []float64 {
	ary := make([]float64, tensor.Len())
	i := 0
	for i = 0; i < tensor.Len(); i++ {
		ary[i] = tensor.FloatVal1D(i)
	}
	return ary
}

// calcF1, using a package that is port of sklearn to ensure identical calculation of F1
func calcF1(sourceTable *etable.Table, predicted, groundTruth string) float64 {
	//predictedAry := TensorToDense(sourceTable.ColByName(predicted))
	//groundTruthAry := TensorToDense(sourceTable.ColByName(groundTruth))
	//Ytrue, Ypred := mat.NewDense(len(predictedAry), 1, predictedAry), mat.NewDense(len(groundTruthAry), 1, groundTruthAry)
	return 0.0 //metrics.F1Score(Ytrue, Ypred, "macro", nil)
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

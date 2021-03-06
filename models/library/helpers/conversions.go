package helpers

import (
	log "github.com/Astera-org/easylog"
	"github.com/Astera-org/obelisk/models/library/metrics"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/goki/gi/gi"
)

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

// WriteActionHistory stores a history of action patterns, removes the last most pattern (since conversion is happening in fworld, and last action won't have a corresponding class)
func WriteTableNLength(sourceTable *etable.Table, filename gi.FileName, numRows int) {
	if sourceTable.Rows == 0 {
		log.Warn("table  is empty")
	} else {
		sourceTable.SetNumRows(numRows) //skip last one, since no action correspondance
		sourceTable.SaveCSV(filename, ',', true)
	}
}

// ActionF1Score calculates the F1 score for a given action and expected action, the last most prediction is not logged, so skip it
func ActionF1Score(sourceTable *etable.Table, predictedName, groundTruthName string, classes []int32) float64 {

	if sourceTable.Rows == 0 {
		log.Warn("ActionHistory is empty, check if discrete actions are coming from Fworld")
		return -1.0
	} else {
		predictTensor := (sourceTable.ColByName(predictedName))
		groundTensor := sourceTable.ColByName(groundTruthName)
		predicted := TensorToAryInt(&predictTensor)
		groundtruth := TensorToAryInt(&groundTensor)
		return metrics.F1ScoreMacro(predicted[:len(predicted)-1], groundtruth[:len(groundtruth)-1], classes) //skip last one, since no action correspondance
	}
}

//	ActionKL calculates kldivergence for action tbale but explicitely skips last most trial where the predicted action may not have been assigned
func ActionKL(sourceTable *etable.Table, predictedCol, expectedCol string, numRows int) float64 {
	if sourceTable.Rows == 0 {
		log.Warn("ActionHistory is empty, check if discrete actions are coming from Fworld")
		return -1.0
	} else {
		predictTensor := (sourceTable.ColByName(predictedCol))
		groundTensor := sourceTable.ColByName(expectedCol)
		predicted := TensorToAryInt(&predictTensor)
		groundtruth := TensorToAryInt(&groundTensor)
		return metrics.KLDivergence(predicted[:numRows], groundtruth[:numRows])
	}
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

func WrapperTensorFloat(value float64) etensor.Tensor {
	predictedTensor := etensor.New(etensor.FLOAT64, []int{1.0}, nil, nil)
	predictedTensor.SetFloat1D(0, float64(value))
	return predictedTensor
}
func WrapperTensorString(value string) etensor.Tensor {
	predictedTensor := etensor.New(etensor.STRING, []int{1}, nil, nil)
	predictedTensor.SetString1D(0, value)

	return predictedTensor
}

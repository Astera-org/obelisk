package metrics

import (
	"testing"
)

func TestMetrics(t *testing.T) {
	predicted := []int32{0, 0, 1, 1, 2}
	groundTruth := []int32{2, 1, 1, 1, 2}

	f1 := (F1ScoreMacro(predicted, groundTruth, []int32{0, 1, 2}))
	recall := Recall(predicted, groundTruth, 1)
	precision := (Precision(predicted, groundTruth, 2))

	if f1 < .4 {
		t.Error("F1ScoreMacro failed, should have gotten around .48 avg")
	}
	if precision != 1.0 {
		t.Error("Precision failed, should have been 1.0")
	}
	if recall < .6 {
		t.Error("recall failed, should have been aprox .66")
	}
}

func TestKL(t *testing.T) {
	//verified kl results when compared to scipy kl_div using distribution p ={.25, .75} q = {.75,. 25}
	groundTruth := []int32{1, 2, 2, 2} //.25, .75
	predict := []int32{1, 1, 1, 2}     //.75, .25

	kl := KLDivergence(predict, groundTruth)
	if kl < .54 {
		t.Error("KLDivergence failed, should have been aprox .55")
	}
	if kl > .55 {
		t.Error("KLDivergence failed, should have been aprox .55")
	}
}

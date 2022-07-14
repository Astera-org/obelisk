package metrics

import "github.com/goki/mat32"

// Precision returns a precision for a given class
func Precision(predicted, groundtruth []int32, relevantClass int32) float64 {
	var tp, fp int32
	predictedExists := false
	for i := 0; i < len(predicted); i++ {
		if predicted[i] == relevantClass {
			predictedExists = true
			if predicted[i] == groundtruth[i] {
				tp++
			} else if predicted[i] != groundtruth[i] {
				fp++
			}
		}
	}
	if predictedExists == false {
		return -1.0 // -1.0 is for class does not exist in predictions
	}
	if tp == 0 && fp == 0 {
		return 0.0
	}
	precision := float64(tp) / float64(tp+fp)

	return precision
}

// Recall returns a recall for a given class
func Recall(predicted, groundtruth []int32, relevantClass int32) float64 {
	tp := 0
	total := 0
	for i := 0; i < len(predicted); i++ {
		if predicted[i] == relevantClass {
			if predicted[i] == groundtruth[i] {
				tp++
			}
		}
		if groundtruth[i] == relevantClass {
			total++
		}
	}
	if total == 0 {
		return -1.0 // -1.0 is for class does not exist
	}
	if tp == 0 && total == 0 {
		return 0.0
	}
	recall := float64(tp) / float64(total)

	return recall
}

// F1Score returns a F1score for a given class
func F1Score(predicted, groundtruth []int32, relevantClass int32) float64 {
	precision := Precision(predicted, groundtruth, relevantClass)
	recall := Recall(predicted, groundtruth, relevantClass)
	if recall == -1.0 || precision == -1.0 {
		return -1.0
	}
	if precision == 0 && recall == 0 {
		return 0.0
	}
	f1 := 2 * precision * recall / (precision + recall)

	return f1
}

// F1ScoreMacro returns a weighted F1score so undercounted classes are not penalized
func F1ScoreMacro(predicted, groundtruth, classes []int32) float64 {
	var f1 float64
	for i := 0; i < len(classes); i++ {
		val := F1Score(predicted, groundtruth, classes[i])
		if val != -1.0 {
			f1 += val
		}
	}
	f1 /= float64(len(classes))
	return f1
}

func CalculateCounts(ary []int32, normalize bool) map[int]float64 {
	amounts := make(map[int]float64)
	total := 0.0
	for _, amount := range ary {
		amounts[int(amount)] += 1.0
		total++
	}
	if normalize == true {
		for name := range amounts {
			amounts[name] = amounts[name] / total
		}
	}
	return amounts
}

// alignCounts ensures that the counts are aligned, i.e. that the same keys exist in both maps
func alignCounts(predicted, actual map[int]float64) (map[int]float64, map[int]float64) {
	for rowName := range predicted {
		_, exists := actual[rowName]
		if exists == false {
			actual[rowName] = 0
		}
	}
	for rowName := range actual {
		_, exists := predicted[rowName]
		if exists == false {
			predicted[rowName] = 0
		}
	}
	return predicted, actual
}

// KLDivergegence calculates kl divergence given raw predictions and actual values
func KLDivergence(predicted, groundtruth []int32) float64 {
	predictedCounts := CalculateCounts(predicted, true)
	groundtruthCounts := CalculateCounts(groundtruth, true)
	return KLDivergeDistributions(predictedCounts, groundtruthCounts)

}

// KLDivergeDistributions calculates kl divergence given two distributions
func KLDivergeDistributions(predDistribution, actualDistribution map[int]float64) float64 {
	predictedCounts, groundtruthCounts := alignCounts(predDistribution, actualDistribution)
	diverge := 0.0
	for name, p := range groundtruthCounts {
		q, _ := predictedCounts[name]
		// handle nan issues
		if q == 1.0 {
			q = .9999
		} else if q == 0.0 {
			q = .0001
		}
		if p == 0.0 {
			p = .0001
		}

		logpq := mat32.Log(float32(p / q))
		plogpq := p * float64(logpq)
		diverge += plogpq
	}
	return diverge

}

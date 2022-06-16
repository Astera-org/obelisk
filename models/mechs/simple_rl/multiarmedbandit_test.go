package main

import (
	"math"
	"math/rand"
	"testing"
	_ "testing"
)

func TestNaiveMultiArmedBandit(t *testing.T) {
	score := make([]float32, 2)
	visits := make([]float32, 2)
	probs := []float32{.7, .9}
	multiarmedBandit, _ := (&NaiveMultiArmedBandit{}).NewUniformPayout(probs)

	pullArms := 500

	for i := 0; i < pullArms; i++ { //just to show correct sampling
		uniform_choice := rand.Intn(len(score))
		score[uniform_choice] += multiarmedBandit.PullArm(uniform_choice)
		visits[uniform_choice] += 1
	}

	for i := 0; i < len(score); i++ {
		distance := math.Abs(float64(probs[i] - (score[i] / visits[i])))
		if distance > .03 {
			t.Fatalf("calculated payoff is greater than expected result: %f ", distance)
		}
	}
}

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

func TestSampling(t *testing.T) {
	world, _ := (&NaiveMultiArmedBandit{}).New([]float32{.25, .01}, []float32{1, 100})
	agent := (&EpsilonGreedy{}).New(*world, .5)

	pullArms := 5000
	for i := 0; i < pullArms; i++ {
		actionTaken := agent.SelectAction()
		world.StepWorld(actionTaken, false)
		agent.UpdateStatistics((*world))
	}

	estimatedRatio := []float32{agent.payoutHistory[0] / float32(agent.visitCounts[0]), agent.payoutHistory[1] / float32(agent.visitCounts[1])}
	for i := 0; i < len(estimatedRatio); i++ {
		distance := math.Abs(float64(world.utility[i] - estimatedRatio[i]))
		if distance > .2 {
			t.Fatalf("calculated payoff is greater than expected result: %f ", distance)
		}
	}

}

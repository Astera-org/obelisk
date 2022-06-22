package main

import (
	"github.com/emer/emergent/agent"
	"math/rand"
)

// EpisolonGreedy is a naive way to solve multi-armed bandit problem, pick current best, but sometimes sample others
type EpsilonGreedy struct {
	epsilon       float32
	payoutHistory []float32
	visitCounts   []int
}

func (greedy *EpsilonGreedy) New(bandit NaiveMultiArmedBandit, epsilon float32) *EpsilonGreedy {
	greedy.epsilon = epsilon
	greedy.payoutHistory = make([]float32, len(bandit.utility))
	greedy.visitCounts = make([]int, len(bandit.utility))
	return greedy
}

func (greedy EpsilonGreedy) bestPayout() (float32, int) {
	best := float32(0.0)
	bestIndex := 0
	for i, val := range greedy.payoutHistory {
		if greedy.visitCounts[i] != 0 {
			score := (val / float32(greedy.visitCounts[i]))
			if score > best {
				bestIndex = i
				best = score
			}
		}
	}
	return best, bestIndex

}
func (greedy EpsilonGreedy) SelectAction() map[string]agent.Action {
	score, index := greedy.bestPayout()
	if score == 0.0 {
		index = int(rand.Float32() * float32(len(greedy.payoutHistory)))
	} else if rand.Float32() < greedy.epsilon { // sample less common choice
		index = int(rand.Float32() * float32(len(greedy.payoutHistory)))
	}
	action := map[string]agent.Action{}
	action["ArmPulled"] = agent.Action{DiscreteOption: index}
	return action
}

//UpdateStatistics updates the payoutHistory and visitCounts based on action and reward
func (greedy *EpsilonGreedy) UpdateStatistics(bandit NaiveMultiArmedBandit) {
	actionChosen := bandit.Observe("ArmPulled")
	reward := bandit.Observe("ObservedReward")

	//argmax of actionChosen
	maxIndex := 0
	arrayLength := len(greedy.payoutHistory)
	for i := 0; i < arrayLength; i++ {
		if actionChosen.FloatVal1D(i) > actionChosen.FloatVal1D(maxIndex) {
			maxIndex = i
		}
	}
	//increment visitCounts and payoutHistory
	greedy.visitCounts[maxIndex] += 1
	greedy.payoutHistory[maxIndex] += float32(reward.FloatVal1D(maxIndex))
}

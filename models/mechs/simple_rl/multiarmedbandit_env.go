package m_armedbandit

import (
	"errors"
	"github.com/emer/emergent/agent"
)

//find empirical data to characterize biological 'solutions'
//https://doi.org/10.1016/j.jtbi.2019.02.002
//Upper Confidence Bounds, Hoeffding's Inequality, UCB, Bayesian UCB, THompson Sampling

//NaiveMultiArmedBandit is a simple implementation of multiarm bandit problem where arms given variable rewards, they are independent, and return a discrete and consistent payout according to some probability (so no std)
type NaiveMultiArmedBandit struct {
	agent.WorldInterface

	armProbs     []float32 //prob of it returning that value
	armPayout    []float32 //how much you're gonna get
	utility      []float32 //true expected return
	runningScore []float32 //a running count of the score of each turn
}

func (mab *NaiveMultiArmedBandit) New(armProbs, armPayout []float32) (*NaiveMultiArmedBandit, error) {
	if len(armPayout) != len(armProbs) {
		return nil, errors.New("Mismatched length")
	}
	mab.armProbs = armProbs
	mab.armPayout = armPayout
	utility := []float32{}
	for i := 0; i < len(armPayout); i++ {
		utility = append(utility, mab.armPayout[i]*mab.armProbs[i])
	}
	mab.utility = utility
	mab.runningScore = make([]float32, len(mab.armProbs))
	return mab, nil
}

//assume you get the same payout
func (mab *NaiveMultiArmedBandit) NewFlatPayout(armProbs []float32) (*NaiveMultiArmedBandit, error) {
	payouts := make([]float32, len(armProbs))
	for i := 0; i < len(payouts); i++ {
		payouts[i] = 1.0
	}
	return mab.New(armProbs, payouts)
}

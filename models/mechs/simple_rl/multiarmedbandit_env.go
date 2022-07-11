package main

import (
	"errors"
	"fmt"
	"github.com/Astera-org/obelisk/models/agent"
	"github.com/emer/etable/etensor"
	"math/rand"
)

//todo add latent variables that are coupled to multibandit problem
//todo add continous sampling instead of discrete values
//find empirical data to characterize biological 'solutions'
//https://doi.org/10.1016/j.jtbi.2019.02.002
//Upper Confidence Bounds, Hoeffding's Inequality, UCB, Bayesian UCB, THompson Sampling

//NaiveMultiArmedBandit is a simple implementation of multiarm bandit problem where arms given variable rewards, they are independent, and return a discrete and consistent payout according to some probability (so no std)
type NaiveMultiArmedBandit struct {
	agent.WorldInterface

	armProbs  []float32 //prob of it returning that value
	armPayout []float32 //how much you're gonna get
	utility   []float32 //true expected return

	lastAction int
	lastReward float32
}

func (mab *NaiveMultiArmedBandit) InitWorld(details map[string]string) (
	actionSpace map[string]agent.SpaceSpec,
	observationSpace map[string]agent.SpaceSpec) {

	discreteActions := agent.SpaceSpec{
		ContinuousShape: []int{len(mab.armPayout)},
		Min:             0.0,
		Max:             1.0,
	}

	reward := agent.SpaceSpec{
		ContinuousShape: []int{len(mab.armPayout)},
		Min:             0.0,
		Max:             10.0,
	}
	return map[string]agent.SpaceSpec{"ArmPulled": discreteActions},
		map[string]agent.SpaceSpec{"ArmPulled": reward, "ObservedReward": discreteActions}
}

// StepWorld steps the index of the current pattern.
func (mab *NaiveMultiArmedBandit) StepWorld(actions map[string]agent.Action, agentDone bool) (worldDone bool, debug string) {
	whichAction := actions["ArmPulled"].DiscreteOption
	mab.lastAction = whichAction
	mab.lastReward = mab.PullArm(whichAction)
	return false, ""
}

// Observe returns an observation from the cache.
func (mab *NaiveMultiArmedBandit) Observe(name string) etensor.Tensor {

	if mab.lastAction == -1 {
		if name == "ObservedReward" {
			tensor := etensor.New(etensor.FLOAT32, []int{len(mab.armPayout)}, nil, []string{"observedReward"})
			tensor.SetFloat1D(0, 0)
			return tensor
		} else if name == "ArmPulled" {
			tensor := etensor.New(etensor.FLOAT32, []int{len(mab.armPayout)}, nil, []string{"armPulled"})
			tensor.SetFloat1D(0, float64(1.0))
			return tensor
		}
	} else {
		if name == "ObservedReward" {
			tensor := etensor.New(etensor.FLOAT32, []int{len(mab.armPayout)}, nil, []string{"observedReward"})
			tensor.SetFloat1D(mab.lastAction, float64(mab.lastReward))
			return tensor
		} else if name == "ArmPulled" {
			tensor := etensor.New(etensor.FLOAT32, []int{len(mab.armPayout)}, nil, []string{"armPulled"})
			tensor.SetFloat1D(mab.lastAction, float64(1.0))
			return tensor
		}
	}
	return nil
}

func (mab *NaiveMultiArmedBandit) New(armProbs, armPayout []float32) (*NaiveMultiArmedBandit, error) {
	mab.lastAction = -1
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
	return mab, nil
}

//assume you get the same payout
func (mab *NaiveMultiArmedBandit) NewUniformPayout(armProbs []float32) (*NaiveMultiArmedBandit, error) {
	payouts := make([]float32, len(armProbs))
	for i := 0; i < len(payouts); i++ {
		payouts[i] = 1.0
	}
	return mab.New(armProbs, payouts)
}

func (mab NaiveMultiArmedBandit) PullArm(whichArm int) float32 {
	probs := rand.Float32()
	threshold := mab.armProbs[whichArm]
	if probs < threshold {
		return mab.armPayout[whichArm]
	}
	return 0.0
}

func main() {
	world, _ := (&NaiveMultiArmedBandit{}).NewUniformPayout([]float32{0, 1})
	world.Observe("Input")

	action := map[string]agent.Action{}
	action["ArmPulled"] = agent.Action{DiscreteOption: 1}
	world.StepWorld(action, false)
	fmt.Println(world.Observe("ObservedReward"))
	fmt.Println(world.Observe("ArmPulled"))
}

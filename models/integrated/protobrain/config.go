package main

import (
	"fmt"

	"github.com/BurntSushi/toml"
)

// LATER move parts of this to a separate package

type Config struct {
	GUI           bool
	PROFILE       bool
	WORKER        bool
	LIFETIME      int32 // how many times the world will call step before we exit
	INTERNAL_PORT int32
	TESTFILE      string // location of sample data to test the neural network, and performance characteristics

	//network hyperparameters
	ACTAVG_INIT     float64 //0.06 7 [.05 .15] "Layer.Inhib.ActAvg.Init"
	LRATE           float64 //.1 scales the amount of weight change
	LayerDecayAct   float64 //.6 8 [0 - 1] "Layer.Act.Decay.Act"
	LayerDecayGlong float64 //.2 "Layer.Act.Decay.Glong"  8 [0 - 1]
	LayerClampGe    float64 //"1.0", // 4 Probs fine  [.6 1.5]

	//Specific to Proto Layers

	LayerTRCDriveScale float64 //TRCLAYER  "0.15", // 10  .3 - .05

	SMALayerActNoiseGe float64 //"Layer.Act.Noise.Ge"` "0.001", //10 .0005 - .01
	SMALayerActNoiseGi float64 //"Layer.Act.Noise.Gi"` "0.001", //10 .0005 - .01

	//Projections

	PrjnSWtAdaptLrate    //string `json:"Prjn.SWt.Adapt.Lrate"`  "0.001", // 8 [.0001 - .01]  0.001 seems to work fine, but .001 maybe more reliable
	PrjnSWtAdaptDreamVar //string `json:"Prjn.SWt.Adapt.DreamVar"` "0.01",  // 8 [0 - .05]  0.01 is just tolerable

	PrjnLearnXCalPThrMin //string `json:"Prjn.Learn.XCal.PThrMin"` // 8 [0.01 - .1]  .05 > .01 for PCA for SynSpk, bad for NeurSpk
	PrjnLearnXCalLrnThr  //string `json:"Prjn.Learn.XCal.LrnThr"` // 8 [0.01 - .1] .05 > .01 here but not smaller nets -- should match NeurCa.LrnThr 0.05 also good

	BackPrjnScaleRel //string `json:"Prjn.PrjnScale.Rel"` //10 [.05 - .3]

	CTBackPrjnScaleRel //string `json:"Prjn.PrjnScale.Rel"` "0.2", //10 [.05 - .3] 0.2 > 0.5 - .ctback ad acttoct should have shared hyperparam

	InhibPrjnLearnLrateBase //string `json:"Prjn.Learn.Lrate.Base"` "0.001", // 9 [.0001 - .01] .0001 > .001 -- slower better!

	InhibPrjnScaleAbs //string `json:"Prjn.PrjnScale.Abs"`  "0.3", // 9 [0 - 0.5] .1 = .2, slower blowup

	LateralPrjnScaleRel //string `json:"Prjn.PrjnScale.Rel"` "0.02", // .02 > .05 == .01 > .1  -- very minor diffs on TE cat

	FmPulPrjnScaleRel // string `json:"Prjn.PrjnScale.Rel"` "0.1", //10 [.05 - .3] .1 > .2
}

func (config *Config) Load() {

	config.setDefaults()
	_, err := toml.DecodeFile("package/default.cfg", &config)
	if err != nil {
		fmt.Println(err)
		fmt.Println("default.cfg not found")
	}
	_, err = toml.DecodeFile("protobrain.cfg", &config)
	if err != nil {
		fmt.Println(err)
		fmt.Println("protobrain.cfg not found")
	}
}

func (config *Config) setDefaults() {
	config.GUI = true
	config.PROFILE = false
	config.WORKER = false
	config.LIFETIME = 100
	config.INTERNAL_PORT = 9090
	config.TESTFILE = "testdata.csv"

	config.LRATE = .1
	config.ACTAVG_INIT = 0.06
}

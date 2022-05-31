package main

import (
	"fmt"
	"testing"

	"github.com/Astera-org/models/library/sim"
	"github.com/emer/axon/axon"
	"github.com/emer/etable/etable"
)

/*
func TestParamAssignment(t *testing.T) {
	//go test -run TestParamAssignment mechs/one2many/*.go
	ss := One2Sim{}
	ss.Pats = &etable.Table{}
	fmt.Println(ss.CmdArgs)
	ss.ApplyHyperFromCMD("../../hyperparams.json")
	//t.Errorf("this should fail")
	//ss.Params.SetMsg = ss.CmdArgs.LogSetParams
	//ss.Params.SetAll()
	// NOTE uncomment following to see the compiled hyper params
	// fmt.Println(ss.Params.NetHypers.JSONString())
	//ss.NewRun()
}*/

func TestConfigNet(t *testing.T) {
	simo := sim.Sim{}
	neto := axon.Network{}
	// TODO params are not set at this point -- triggers warning in confignet
	ConfigNet(&simo, &neto)
	if len(neto.Layers) != 4 {
		t.Errorf("Expected network to be configured differently")
	}
}

func TestConfigPats(t *testing.T) {
	ss := One2Sim{}
	ss.Pats = &etable.Table{}
	ss.NInputs = 5
	ss.NOutputs = 2
	ConfigPats(&ss)
	if ss.Pats.Rows < 10 {
		fmt.Println(*ss.Pats)
		t.Errorf("Expected more patterns than that!")
	}
}

func SimpleTest(t *testing.T) {
	for i := 0; i < 100; i++ {
		fmt.Println("got to {}", i)
	}
	if 9 > 100 {
		t.Errorf("oh no!")
	}
}

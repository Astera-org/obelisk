// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"github.com/emer/emergent/params"
)

func createParams(gconfig Config) params.Sets {

	lrate := fmt.Sprintf("%f", gconfig.LRATE)
	act_init := fmt.Sprintf("%f", gconfig.ACTAVG_INIT)

	layerDecayAct := fmt.Sprintf("%f", gconfig.LayerDecayAct)
	layerDecayGlong := fmt.Sprintf("%f", gconfig.LayerDecayGlong)
	layerClampGe := fmt.Sprintf("%f", gconfig.LayerClampGe)
	//Specific to Proto Layers
	layerTRCDriveScale := fmt.Sprintf("%f", gconfig.LayerTRCDriveScale)
	smaLayerActNoiseGe := fmt.Sprintf("%f", gconfig.SMALayerActNoiseGe)
	smaLayerActNoiseGi := fmt.Sprintf("%f", gconfig.SMALayerActNoiseGi)
	//Projections
	prjnSWtAdaptLrate := fmt.Sprintf("%f", gconfig.PrjnSWtAdaptLrate)
	prjnSWtAdaptDreamVar := fmt.Sprintf("%f", gconfig.PrjnSWtAdaptDreamVar)
	prjnLearnXCalPThrMin := fmt.Sprintf("%f", gconfig.PrjnLearnXCalPThrMin)
	prjnLearnXCalLrnThr := fmt.Sprintf("%f", gconfig.PrjnLearnXCalLrnThr)
	backPrjnScaleRel := fmt.Sprintf("%f", gconfig.BackPrjnScaleRel)
	ctBackPrjnScaleRel := fmt.Sprintf("%f", gconfig.CTBackPrjnScaleRel)
	inhibPrjnLearnLrateBase := fmt.Sprintf("%f", gconfig.InhibPrjnLearnLrateBase)
	inhibPrjnScaleAbs := fmt.Sprintf("%f", gconfig.InhibPrjnScaleAbs)
	lateralPrjnScaleRel := fmt.Sprintf("%f", gconfig.LateralPrjnScaleRel)
	fmPulPrjnScaleRel := fmt.Sprintf("%f", gconfig.FmPulPrjnScaleRel)

	// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
	// selected to apply on top of that
	var ParamSets = params.Sets{
		{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
			"Network": &params.Sheet{
				{Sel: "Layer", Desc: "using default 1 inhib for hidden layers",
					Params: params.Params{
						"Layer.Inhib.ActAvg.Init": act_init, //7 [.05 - .15] model dependent   sets an expected activity layer for that level, used to rescale the activity
						"Layer.Inhib.Layer.Gi":    "1.1",    //1 not important
						"Layer.Inhib.Pool.FFEx0":  "0.15",   //
						"Layer.Inhib.Pool.FFEx":   "0.02",   // .05 for lvis
						"Layer.Inhib.Layer.FFEx0": "0.15",
						"Layer.Inhib.Layer.FFEx":  "0.02", //
						//"Layer.Act.Gbar.L":        "0.2", // NOT IMPORTANT
						"Layer.Act.Decay.Act":   layerDecayAct,   // 8 [0 - 1]
						"Layer.Act.Decay.Glong": layerDecayGlong, // 8 [0 - 1]
						"Layer.Act.Clamp.Ge":    layerClampGe,    // 4 Probs fine  [.6 1.5]
						"Layer.Inhib.Pool.Gi":   "1.1",
					}},
				{Sel: ".Hidden", Desc: "noise? sub-pools",
					Params: params.Params{
						"Layer.Inhib.ActAvg.Init":    "0.06",
						"Layer.Inhib.ActAvg.AdaptGi": "false", // no!
						"Layer.Inhib.Pool.On":        "true",
						"Layer.Inhib.Layer.On":       "true", // full layer
					}},
				{Sel: ".CT", Desc: "corticothalamic context",
					Params: params.Params{
						"Layer.Inhib.ActAvg.Init": "0.06",
						"Layer.CtxtGeGain":        "0.2", //10, [.1 - .3]
						"Layer.Inhib.Pool.On":     "true",
						"Layer.Inhib.Layer.On":    "true",
						"Layer.Act.KNa.On":        "true",
						"Layer.Act.NMDA.Gbar":     "0.03", // larger not better
						"Layer.Act.GABAB.Gbar":    "0.2",
						"Layer.Act.Decay.Act":     "0.0", // 0 best in other models
						"Layer.Act.Decay.Glong":   "0.0",
					}},
				{Sel: "TRCLayer", Desc: "",
					Params: params.Params{ //LEAVE TRCLAYER alone except for trc.drivescale
						"Layer.TRC.DriveScale":   layerTRCDriveScale, // 10  .3 - .05
						"Layer.Act.Decay.Act":    "0.5",              //LEAVE THIS ON
						"Layer.Act.Decay.Glong":  "1",                // clear long
						"Layer.Inhib.Pool.FFEx":  "0.0",
						"Layer.Inhib.Layer.FFEx": "0.0",
					}},
				{Sel: ".Depth", Desc: "depth layers use pool inhibition only",
					Params: params.Params{
						"Layer.Inhib.ActAvg.Init": "0.08",
						"Layer.Inhib.Layer.On":    "true",
						"Layer.Inhib.Pool.On":     "false",
						"Layer.Inhib.Layer.Gi":    "0.8", //LEAVE ALONE
						"Layer.Inhib.Pool.Gi":     "0.8", //LEAVE ALONE
						"Layer.Inhib.Pool.FFEx":   "0.0", //LEAVE ALONE
						"Layer.Inhib.Layer.FFEx":  "0.0", //LEAVE ALONE
					}},
				{Sel: ".Fovea", Desc: "fovea has both",
					Params: params.Params{ //LEAVE ALONE
						"Layer.Inhib.ActAvg.Init": "0.15",
						"Layer.Inhib.Layer.On":    "true", // layer too
						"Layer.Inhib.Layer.Gi":    "1",
						"Layer.Inhib.Pool.On":     "true",
						"Layer.Inhib.Pool.Gi":     "1",
						"Layer.Inhib.Pool.FFEx":   "0.0",
						"Layer.Inhib.Layer.FFEx":  "0.0",
					}},
				{Sel: ".S1S", Desc: "lower inhib, higher act",
					Params: params.Params{ //LEAVE ALONE
						"Layer.Inhib.Layer.Gi":    "1", // some weaker global inhib
						"Layer.Inhib.Pool.On":     "true",
						"Layer.Inhib.Pool.Gi":     "0.8", // weaker
						"Layer.Inhib.ActAvg.Init": "0.2",
					}},
				{Sel: ".S1V", Desc: "lower inhib, higher act",
					Params: params.Params{
						"Layer.Inhib.Layer.On": "true",
						"Layer.Inhib.Pool.On":  "false",
					}},
				{Sel: ".Ins", Desc: "pools",
					Params: params.Params{
						"Layer.Inhib.Pool.On": "true",
					}},
				{Sel: ".M1", Desc: "",
					Params: params.Params{
						"Layer.Inhib.ActAvg.Init": "0.12",
					}},
				{Sel: ".MSTd", Desc: "",
					Params: params.Params{
						"Layer.Inhib.Layer.On":    "true",
						"Layer.Inhib.Pool.On":     "true",
						"Layer.Inhib.ActAvg.Init": "0.03",
						"Layer.Inhib.Pool.FFEx":   "0.02", //
						"Layer.Inhib.Layer.FFEx":  "0.02",
					}},
				{Sel: "#MSTdCT", Desc: "",
					Params: params.Params{
						// "Layer.Inhib.Layer.Gi": "1.1",
						// "Layer.Inhib.Pool.Gi":  "1.1",
						// "Layer.Inhib.Pool.FFEx":   "0.08", // .05 for lvis
						// "Layer.Inhib.Layer.FFEx":  "0.08", // .05 best so far
					}},
				{Sel: ".cIPL", Desc: "cIPL general",
					Params: params.Params{
						"Layer.Inhib.Layer.On": "true",
						"Layer.Inhib.Pool.On":  "true",
					}},
				{Sel: ".PCC", Desc: "PCC general",
					Params: params.Params{
						"Layer.Inhib.Layer.On": "true",
						"Layer.Inhib.Pool.On":  "true",
					}},
				{Sel: "#V2WdP", Desc: "weaker inhibition for pulvinar",
					Params: params.Params{ //LEAVE ALONE
						"Layer.Inhib.Layer.Gi": "0.8",
						"Layer.Inhib.Pool.Gi":  "0.8", // not used
					}},
				{Sel: "#MSTdP", Desc: "weaker inhibition for pulvinar",
					Params: params.Params{ //LEAVE ALONE
						"Layer.Inhib.Layer.Gi": "0.9", // 0.8 > 0.9
						"Layer.Inhib.Pool.Gi":  "0.9",
					}},
				{Sel: "#cIPLP", Desc: "weaker inhibition for pulvinar",
					Params: params.Params{ //LEAVE ALONE
						"Layer.Inhib.Layer.Gi": "0.9",
						"Layer.Inhib.Pool.Gi":  "0.9",
					}},
				{Sel: ".SMA", Desc: "",
					Params: params.Params{ //LEAVE ALONE
						"Layer.Inhib.ActAvg.Init": "0.12",
						"Layer.Inhib.Pool.On":     "false",
					}},
				{Sel: "#SMA", Desc: "",
					Params: params.Params{
						"Layer.Act.Noise.On": "true",
						"Layer.Act.Noise.Ge": smaLayerActNoiseGe, //10 .0005 - .01
						"Layer.Act.Noise.Gi": smaLayerActNoiseGi, //10 .0005 - .01
					}},
				{Sel: "#SMAP", Desc: "pulv",
					Params: params.Params{
						"Layer.Inhib.Pool.On":     "true", // independent pathways
						"Layer.Inhib.ActAvg.Init": "0.1",
					}},
				{Sel: "#Act", Desc: "",
					Params: params.Params{ //LEAVE ALONE
						"Layer.Inhib.ActAvg.Init": "0.12",
					}},
				{Sel: "#VL", Desc: "VL regular inhib", //LEAVE ALONE
					Params: params.Params{
						"Layer.Inhib.Layer.Gi":   "0.8", //10 [0.5 - 1.0]
						"Layer.Inhib.Pool.FFEx":  "0.0",
						"Layer.Inhib.Layer.FFEx": "0.0",
					}},
				{Sel: "#M1", Desc: "noise!?", //LEAVE ALONE
					Params: params.Params{
						"Layer.Inhib.ActAvg.Init": "0.12",
						"Layer.Inhib.Layer.Gi":    "1.1", // reg
					}},
				{Sel: "#M1P", Desc: "m1 pulvinar", //LEAVE ALONE
					Params: params.Params{
						"Layer.Inhib.ActAvg.Init": "0.12",
						"Layer.Inhib.Layer.Gi":    "1.0", // weaker pulv
					}},
				{Sel: ".IT", Desc: "reg",
					Params: params.Params{ //LEAVE MOST ALONE
						"Layer.Inhib.ActAvg.Init": "0.12",
						"Layer.Inhib.Pool.On":     "false",
					}},
				//////////////////////////////////////////////////////////
				// Prjns

				{Sel: "Prjn", Desc: "norm and momentum on is critical, wt bal not as much but fine",
					Params: params.Params{
						"Prjn.Learn.Lrate.Base":       lrate,                // 10  [.001, .1, .5]  .01 for SynSpkTheta
						"Prjn.SWt.Adapt.Lrate":        prjnSWtAdaptLrate,    // 8 [.0001 - .01]  0.001 seems to work fine, but .001 maybe more reliable
						"Prjn.SWt.Adapt.DreamVar":     prjnSWtAdaptDreamVar, // 8 [0 - .05]  0.01 is just tolerable
						"Prjn.SWt.Init.SPct":          "1.0",                // .5 ok here, 1 best for larger nets: objrec, lvis
						"Prjn.Learn.KinaseCa.SpikeG":  "12",                 // 12 matches theta exactly, higher dwtavg but ok
						"Prjn.Learn.KinaseCa.NMDAG":   "1",
						"Prjn.Learn.KinaseCa.Rule":    "SynSpkTheta", //REMOVE
						"Prjn.Learn.KinaseCa.MTau":    "5",           // 5 > 10 test more
						"Prjn.Learn.KinaseCa.UpdtThr": "0.05",        // 0.05 -- was LrnThr
						"Prjn.Learn.XCal.On":          "true",
						"Prjn.Learn.XCal.PThrMin":     prjnLearnXCalPThrMin, // 8 [0.01 - .1]  .05 > .01 for PCA for SynSpk, bad for NeurSpk
						"Prjn.Learn.XCal.LrnThr":      prjnLearnXCalLrnThr,  // 8 [0.01 - .1] .05 > .01 here but not smaller nets -- should match NeurCa.LrnThr 0.05 also good
					}},
				{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
					Params: params.Params{
						"Prjn.PrjnScale.Rel": backPrjnScaleRel, //10 [.05 - .3]
					}},
				{Sel: ".CTBack", Desc: "deep top-down -- stronger",
					Params: params.Params{
						"Prjn.PrjnScale.Rel": ctBackPrjnScaleRel, //10 [.05 - .3] 0.2 > 0.5 - .ctback ad acttoct should have shared hyperparam
					}},
				{Sel: ".ActToCT", Desc: "weaker",
					Params: params.Params{
						"Prjn.PrjnScale.Rel": "0.2",
					}},
				{Sel: ".Inhib", Desc: "inhibitory projection",
					Params: params.Params{
						"Prjn.Learn.Learn":      "true",                  // learned decorrel is good
						"Prjn.Learn.Lrate.Base": inhibPrjnLearnLrateBase, // 9 [.0001 - .01] .0001 > .001 -- slower better!
						"Prjn.SWt.Init.Var":     "0.0",
						"Prjn.SWt.Init.Mean":    "0.1",
						"Prjn.SWt.Init.Sym":     "false",
						"Prjn.SWt.Adapt.On":     "false",
						"Prjn.PrjnScale.Abs":    inhibPrjnScaleAbs, // 9 [0 - 0.5] .1 = .2, slower blowup
						"Prjn.IncGain":          "1",               //TODO randy take a look, ill poke randy  .5 def
					}},
				{Sel: ".Lateral", Desc: "default for lateral -- not using", //DO NOT REMOVE
					Params: params.Params{
						"Prjn.SWt.Init.Sym":  "false",
						"Prjn.SWt.Init.Var":  "0",
						"Prjn.PrjnScale.Rel": lateralPrjnScaleRel, // .02 > .05 == .01 > .1  -- very minor diffs on TE cat
					}},
				{Sel: ".CTFmSuper", Desc: "CT from main super", //DO NOT REMOVE
					Params: params.Params{
						"Prjn.PrjnScale.Rel": "1", // 0.5 > 0.2
					}},
				{Sel: ".SuperFwd", Desc: "standard superficial forward prjns -- not to output",
					Params: params.Params{ //DO NOT REMOVE
						"Prjn.Com.PFail":    "0.0",   // 0.5 sig worse perf, 0.2 ~= 0.1
						"Prjn.Com.PFailSWt": "false", // try
					}},
				{Sel: ".FmPulv", Desc: "default for pulvinar",
					Params: params.Params{
						"Prjn.PrjnScale.Rel": fmPulPrjnScaleRel, //10 [.05 - .3] .1 > .2
					}},
				{Sel: ".CTSelf", Desc: "CT to CT",
					Params: params.Params{ //DO NOT REMOVE
						"Prjn.PrjnScale.Rel": "0.5", // 0.5 > 0.2
					}},
				{Sel: ".CTToPulv", Desc: "basic main CT to pulivnar -- needs to be stronger -- cons are weak somehow",
					Params: params.Params{ //DO NOT REMOVE
						"Prjn.PrjnScale.Abs": "1.5",
						"Prjn.PrjnScale.Rel": "1",
					}},
				{Sel: ".CTToPulv3", Desc: "even stronger abs",
					Params: params.Params{ //DO NOT REMOVE
						"Prjn.PrjnScale.Abs": "3",
						"Prjn.PrjnScale.Rel": "1",
					}},
				{Sel: ".ToPulv1", Desc: "weaker higher-level pulvinar prjn",
					Params: params.Params{ //DO NOT REMOVE
						"Prjn.PrjnScale.Abs": "1.5",
						"Prjn.PrjnScale.Rel": "0.1",
					}},
				{Sel: ".ToPulv2", Desc: "weaker higher-level pulvinar prjn",
					Params: params.Params{ //DO NOT REMOVE
						"Prjn.PrjnScale.Abs": "1.5",
						"Prjn.PrjnScale.Rel": "0.2",
					}},
				{Sel: ".FwdToPulv", Desc: "feedforward to pulvinar directly",
					Params: params.Params{ //DO NOT REMOVE
						"Prjn.PrjnScale.Rel": "0.1",
					}},
				{Sel: "#ITToITCT", Desc: "IT likes stronger FmSuper",
					Params: params.Params{ //DO NOT REMOVE
						"Prjn.PrjnScale.Rel": "1", // 0.5 > 0.2
					}},
				{Sel: "#LIPToLIPCT", Desc: "LIP likes stronger FmSuper",
					Params: params.Params{ //DO NOT REMOVE
						"Prjn.PrjnScale.Rel": "1", // 0.5 > 0.2
					}},
				{Sel: "#LIPCTToLIPCT", Desc: "LIP likes stronger CTSelf",
					Params: params.Params{ //DO NOT REMOVE
						"Prjn.PrjnScale.Rel": "1", // 0.5 > 0.2
					}},
				{Sel: ".V1SC", Desc: "v1 shortcut",
					Params: params.Params{
						"Prjn.Learn.Lrate.Base": "0.001", //
						"Prjn.PrjnScale.Rel":    "0.5",   // .5 lvis
						"Prjn.SWt.Adapt.On":     "false", // seems better
					}},
			},
		}},
	}
	return ParamSets
}

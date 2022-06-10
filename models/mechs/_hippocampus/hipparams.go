package main

import (
	"github.com/Astera-org/models/library/sim"
	"github.com/emer/emergent/evec"
	"github.com/emer/emergent/params"
)

// HipParams have the hippocampus size and connectivity parameters
type HipParams struct {
	ECSize       evec.Vec2i `desc:"size of EC in terms of overall pools (outer dimension)"`
	ECPool       evec.Vec2i `desc:"size of one EC pool"`
	CA1Pool      evec.Vec2i `desc:"size of one CA1 pool"`
	CA3Size      evec.Vec2i `desc:"size of CA3"`
	DGRatio      float32    `desc:"size of DG / CA3"`
	DGSize       evec.Vec2i `inactive:"+" desc:"size of DG"`
	DGPCon       float32    `desc:"percent connectivity into DG"`
	CA3PCon      float32    `desc:"percent connectivity into CA3"`
	MossyPCon    float32    `desc:"percent connectivity into CA3 from DG"`
	ECPctAct     float32    `desc:"percent activation in EC pool"`
	MossyDel     float32    `desc:"delta in mossy effective strength between minus and plus phase"`
	MossyDelTest float32    `desc:"delta in mossy strength for testing (relative to base param)"`
}

func (hp *HipParams) Defaults() {
	// size
	hp.ECSize.Set(2, 3)
	hp.ECPool.Set(7, 7)
	hp.CA1Pool.Set(15, 15) // using MedHip now
	hp.CA3Size.Set(30, 30) // using MedHip now
	hp.DGRatio = 2.236     // c.f. Ketz et al., 2013

	// TODO This was in Update(). Should it be there?
	hp.DGSize.X = int(float32(hp.CA3Size.X) * hp.DGRatio)
	hp.DGSize.Y = int(float32(hp.CA3Size.Y) * hp.DGRatio)

	// ratio
	hp.DGPCon = 0.25 // .35 is sig worse, .2 learns faster but AB recall is worse
	hp.CA3PCon = 0.25
	hp.MossyPCon = 0.02 // .02 > .05 > .01 (for small net)
	hp.ECPctAct = 0.2

	hp.MossyDel = 3     // 4 > 2 -- best is 4 del on 4 rel baseline
	hp.MossyDelTest = 0 // for rel = 4: 3 > 2 > 0 > 4 -- 4 is very bad -- need a small amount..
}

func (hp *HipParams) Update() {
	hp.DGSize.X = int(float32(hp.CA3Size.X) * hp.DGRatio)
	hp.DGSize.Y = int(float32(hp.CA3Size.Y) * hp.DGRatio)
}

// ConfigParams configure the parameters
func ConfigParams(ss *sim.Sim) {
	ss.Params.AddNetwork(ss.Net)
	ss.Params.AddSim(ss)
	ss.Params.AddNetSize()
	ss.Params.Params = params.Sets{
		{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
			"Network": &params.Sheet{
				{Sel: "Layer", Desc: "generic layer params",
					Params: params.Params{
						"Layer.Act.KNa.On":         "false", // false > true
						"Layer.Learn.TrgAvgAct.On": "false", // true > false?
						"Layer.Learn.RLrate.On":    "false", // no diff..
						"Layer.Act.Gbar.L":         "0.2",   // .2 > .1
						"Layer.Act.Decay.Act":      "1.0",   // 1.0 both is best by far!
						"Layer.Act.Decay.Glong":    "1.0",
						"Layer.Inhib.Pool.Bg":      "0.0",
					}, Hypers: params.Hypers{
						"Layer.Act.Gbar.L": {"StdDev": "0.1"},
					}},
				{Sel: ".EC", Desc: "all EC layers: only pools, no layer-level",
					Params: params.Params{
						"Layer.Learn.TrgAvgAct.On": "false", // def true, not rel?
						"Layer.Learn.RLrate.On":    "false", // def true, too slow?
						"Layer.Inhib.ActAvg.Init":  "0.15",
						"Layer.Inhib.Layer.On":     "false",
						"Layer.Inhib.Layer.Gi":     "0.2", // weak just to keep it from blowing up
						"Layer.Inhib.Pool.Gi":      "1.1",
						"Layer.Inhib.Pool.On":      "true",
					}},
				{Sel: "#ECout", Desc: "all EC layers: only pools, no layer-level",
					Params: params.Params{
						"Layer.Inhib.Pool.Gi": "1.1",
						"Layer.Act.Clamp.Ge":  "0.6",
					}},
				{Sel: "#CA1", Desc: "CA1 only Pools",
					Params: params.Params{
						"Layer.Learn.TrgAvgAct.On": "true",  // actually a bit better
						"Layer.Learn.RLrate.On":    "false", // def true, too slow?
						"Layer.Inhib.ActAvg.Init":  "0.02",
						"Layer.Inhib.Layer.On":     "false",
						"Layer.Inhib.Pool.Gi":      "1.3", // 1.3 > 1.2 > 1.1
						"Layer.Inhib.Pool.On":      "true",
						"Layer.Inhib.Pool.FFEx0":   "1.0", // blowup protection
						"Layer.Inhib.Pool.FFEx":    "0.0",
					}},
				{Sel: "#DG", Desc: "very sparse = high inibhition",
					Params: params.Params{
						"Layer.Inhib.ActAvg.Init": "0.005", // actual .002-3
						"Layer.Inhib.Layer.Gi":    "2.2",   // 2.2 > 2.0 on larger
					}},
				{Sel: "#CA3", Desc: "sparse = high inibhition",
					Params: params.Params{
						"Layer.Inhib.ActAvg.Init": "0.02",
						"Layer.Inhib.Layer.Gi":    "1.8", // 1.8 > 1.6 > 2.0
					}},
				{Sel: "Prjn", Desc: "keeping default params for generic prjns",
					Params: params.Params{
						"Prjn.SWt.Init.SPct": "0.5", // 0.5 == 1.0 > 0.0
						//"Prjn.Learn.KinaseCa.Rule": "NeurSpkTheta",
					}},
				{Sel: ".EcCa1Prjn", Desc: "encoder projections",
					Params: params.Params{
						"Prjn.Learn.Lrate.Base": "0.04", // 0.04 for Axon -- 0.01 for EcCa1
					}, Hypers: params.Hypers{
						"Prjn.Learn.Lrate.Base": {"StdDev": "0.02"},
					}},
				{Sel: ".HippoCHL", Desc: "hippo CHL projections",
					Params: params.Params{
						"Prjn.CHL.Hebb":         "0.05",
						"Prjn.Learn.Lrate.Base": "0.02", // .2 def
					}, Hypers: params.Hypers{
						"Prjn.Learn.Lrate.Base": {"StdDev": "0.01", "Min": "0"},
					}},
				{Sel: ".PPath", Desc: "perforant path, new Dg error-driven EcCa1Prjn prjns",
					Params: params.Params{
						"Prjn.Learn.Lrate.Base": "0.1", // .1 > .04 -- makes a diff
						// moss=4, delta=4, lr=0.2, test = 3 are best
					}},
				{Sel: "#CA1ToECout", Desc: "extra strong from CA1 to ECout",
					Params: params.Params{
						"Prjn.PrjnScale.Abs": "2.0", // 2.0 > 3.0 for larger
					}},
				{Sel: "#ECinToCA3", Desc: "stronger",
					Params: params.Params{
						"Prjn.PrjnScale.Abs": "3.0", // 4.0 > 3.0
					}},
				{Sel: "#ECinToDG", Desc: "DG learning is surprisingly critical: maxed out fast, hebbian works best",
					Params: params.Params{
						"Prjn.Learn.Learn":      "true", // absolutely essential to have on!
						"Prjn.CHL.Hebb":         "0.5",  // .5 > 1 overall
						"Prjn.CHL.SAvgCor":      "0.1",  // .1 > .2 > .3 > .4 ?
						"Prjn.CHL.MinusQ1":      "true", // dg self err?
						"Prjn.Learn.Lrate.Base": "0.01", // 0.01 > 0.04 maybe
					}},
				{Sel: "#InputToECin", Desc: "one-to-one input to EC",
					Params: params.Params{
						"Prjn.Learn.Learn":   "false",
						"Prjn.SWt.Init.Mean": "0.9",
						"Prjn.SWt.Init.Var":  "0.0",
						"Prjn.PrjnScale.Abs": "1.0",
					}},
				{Sel: "#ECoutToECin", Desc: "one-to-one out to in",
					Params: params.Params{
						"Prjn.Learn.Learn":   "false",
						"Prjn.SWt.Init.Mean": "0.9",
						"Prjn.SWt.Init.Var":  "0.01",
						"Prjn.PrjnScale.Rel": "0.5", // 0.5 > 1 (sig worse)
					}},
				{Sel: "#DGToCA3", Desc: "Mossy fibers: strong, non-learning",
					Params: params.Params{
						"Prjn.Learn.Learn":   "false",
						"Prjn.SWt.Init.Mean": "0.9",
						"Prjn.SWt.Init.Var":  "0.01",
						"Prjn.PrjnScale.Rel": "3", // 4 def
					}},
				{Sel: "#CA3ToCA3", Desc: "CA3 recurrent cons",
					Params: params.Params{
						"Prjn.PrjnScale.Rel":    "0.1",  // 0.1 > 0.2 == 0
						"Prjn.Learn.Lrate.Base": "0.04", // 0.1 v.s .04 not much diff
					}},
				{Sel: "#CA3ToCA1", Desc: "Schaffer collaterals -- slower, less hebb",
					Params: params.Params{
						// "Prjn.CHL.Hebb":         "0.01",
						// "Prjn.CHL.SAvgCor":      "0.4",
						"Prjn.Learn.Lrate.Base": "0.1", // 0.1 > 0.04
						"Prjn.PrjnScale.Rel":    "2",   // 2 > 1
					}},
				{Sel: "#ECoutToCA1", Desc: "weaker",
					Params: params.Params{
						"Prjn.PrjnScale.Rel": "1.0", // 1.0 -- try 0.5
					}},
			},
			// NOTE: it is essential not to put Pat / Hip params here, as we have to use Base
			// to initialize the network every time, even if it is a different size.
		}},
		//{Name: "List010", Desc: "list size", Sheets: params.Sheets{
		//	"Pat": &params.Sheet{
		//		{Sel: "Pat", Desc: "pattern params",
		//			Params: params.Params{
		//				"Pat.ListSize": "10",
		//			}},
		//	},
		//}},
		//{Name: "List020", Desc: "list size", Sheets: params.Sheets{
		//	"Pat": &params.Sheet{
		//		{Sel: "Pat", Desc: "pattern params",
		//			Params: params.Params{
		//				"Pat.ListSize": "20",
		//			}},
		//	},
		//}},
		//{Name: "List030", Desc: "list size", Sheets: params.Sheets{
		//	"Pat": &params.Sheet{
		//		{Sel: "Pat", Desc: "pattern params",
		//			Params: params.Params{
		//				"Pat.ListSize": "30",
		//			}},
		//	},
		//}},
		//{Name: "List040", Desc: "list size", Sheets: params.Sheets{
		//	"Pat": &params.Sheet{
		//		{Sel: "Pat", Desc: "pattern params",
		//			Params: params.Params{
		//				"Pat.ListSize": "40",
		//			}},
		//	},
		//}},
		//{Name: "List050", Desc: "list size", Sheets: params.Sheets{
		//	"Pat": &params.Sheet{
		//		{Sel: "Pat", Desc: "pattern params",
		//			Params: params.Params{
		//				"Pat.ListSize": "50",
		//			}},
		//	},
		//}},
		//{Name: "List060", Desc: "list size", Sheets: params.Sheets{
		//	"Pat": &params.Sheet{
		//		{Sel: "Pat", Desc: "pattern params",
		//			Params: params.Params{
		//				"Pat.ListSize": "60",
		//			}},
		//	},
		//}},
		//{Name: "List070", Desc: "list size", Sheets: params.Sheets{
		//	"Pat": &params.Sheet{
		//		{Sel: "Pat", Desc: "pattern params",
		//			Params: params.Params{
		//				"Pat.ListSize": "70",
		//			}},
		//	},
		//}},
		//{Name: "List080", Desc: "list size", Sheets: params.Sheets{
		//	"Pat": &params.Sheet{
		//		{Sel: "Pat", Desc: "pattern params",
		//			Params: params.Params{
		//				"Pat.ListSize": "80",
		//			}},
		//	},
		//}},
		//{Name: "List090", Desc: "list size", Sheets: params.Sheets{
		//	"Pat": &params.Sheet{
		//		{Sel: "Pat", Desc: "pattern params",
		//			Params: params.Params{
		//				"Pat.ListSize": "90",
		//			}},
		//	},
		//}},
		//{Name: "List100", Desc: "list size", Sheets: params.Sheets{
		//	"Pat": &params.Sheet{
		//		{Sel: "Pat", Desc: "pattern params",
		//			Params: params.Params{
		//				"Pat.ListSize": "100",
		//			}},
		//	},
		//}},
		//{Name: "List125", Desc: "list size", Sheets: params.Sheets{
		//	"Pat": &params.Sheet{
		//		{Sel: "Pat", Desc: "pattern params",
		//			Params: params.Params{
		//				"Pat.ListSize": "125",
		//			}},
		//	},
		//}},
		//{Name: "List150", Desc: "list size", Sheets: params.Sheets{
		//	"Pat": &params.Sheet{
		//		{Sel: "Pat", Desc: "pattern params",
		//			Params: params.Params{
		//				"Pat.ListSize": "150",
		//			}},
		//	},
		//}},
		//{Name: "List200", Desc: "list size", Sheets: params.Sheets{
		//	"Pat": &params.Sheet{
		//		{Sel: "Pat", Desc: "pattern params",
		//			Params: params.Params{
		//				"Pat.ListSize": "200",
		//			}},
		//	},
		//}},
		//{Name: "SmallHip", Desc: "hippo size", Sheets: params.Sheets{
		//	"Hip": &params.Sheet{
		//		{Sel: "HipParams", Desc: "hip sizes",
		//			Params: params.Params{
		//				"HipParams.ECPool.Y":  "7",
		//				"HipParams.ECPool.X":  "7",
		//				"HipParams.CA1Pool.Y": "10",
		//				"HipParams.CA1Pool.X": "10",
		//				"HipParams.CA3Size.Y": "20",
		//				"HipParams.CA3Size.X": "20",
		//				"HipParams.DGRatio":   "2.236", // 1.5 before, sqrt(5) aligns with Ketz et al. 2013
		//			}},
		//	},
		//}},
		//{Name: "MedHip", Desc: "hippo size", Sheets: params.Sheets{
		//	"Hip": &params.Sheet{
		//		{Sel: "HipParams", Desc: "hip sizes",
		//			Params: params.Params{
		//				"HipParams.ECPool.Y":  "7",
		//				"HipParams.ECPool.X":  "7",
		//				"HipParams.CA1Pool.Y": "15",
		//				"HipParams.CA1Pool.X": "15",
		//				"HipParams.CA3Size.Y": "30",
		//				"HipParams.CA3Size.X": "30",
		//				"HipParams.DGRatio":   "2.236", // 1.5 before
		//			}},
		//	},
		//}},
		//{Name: "BigHip", Desc: "hippo size", Sheets: params.Sheets{
		//	"Hip": &params.Sheet{
		//		{Sel: "HipParams", Desc: "hip sizes",
		//			Params: params.Params{
		//				"HipParams.ECPool.Y":  "7",
		//				"HipParams.ECPool.X":  "7",
		//				"HipParams.CA1Pool.Y": "20",
		//				"HipParams.CA1Pool.X": "20",
		//				"HipParams.CA3Size.Y": "40",
		//				"HipParams.CA3Size.X": "40",
		//				"HipParams.DGRatio":   "2.236", // 1.5 before
		//			}},
		//	},
		//}},
	}
}

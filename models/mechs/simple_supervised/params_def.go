// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "github.com/emer/emergent/params"

// ParamSets is a set of parameters that have been found through trial and error to work well for this task.
// Base is always applied, and others can be optionally selected to apply on top of that.
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are params that have been found to work well", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "Deviations from the default parameters that are more appropriate for a small network such as this.",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":     "1.2",
					"Layer.Inhib.ActAvg.Init":  "0.04",
					"Layer.Act.VGCC.Gbar":      "0.1",
					"Layer.Act.AK.Gbar":        "1",
					"Layer.Act.NMDA.MgC":       "1.2",
					"Layer.Act.NMDA.Voff":      "0",
					"Layer.Learn.NeurCa.CaMax": "100",
					"Layer.Learn.NeurCa.Decay": "true",
				},
			},
			{Sel: "#Input", Desc: "critical now to specify the activity level",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "0.9",
					"Layer.Inhib.ActAvg.Init": "0.15",
				}},
			{Sel: "#Output", Desc: "output definitely needs lower inhibition -- true for smaller layers in general",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "0.9",
					"Layer.Act.Spike.Tr":      "1",
					"Layer.Inhib.ActAvg.Init": "0.24",
				}},
			{Sel: "Prjn", Desc: "basic projection params",
				Params: params.Params{
					"Prjn.SWt.Adapt.Lrate":      "0.08",
					"Prjn.SWt.Init.SPct":        "0.5",
					"Prjn.Learn.Lrate.Base":     "0.1",
					"Prjn.Learn.KinaseCa.NMDAG": "2",
					"Prjn.Learn.XCal.PThrMin":   "0.01",
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.3",
				}},
		},
	}},
}

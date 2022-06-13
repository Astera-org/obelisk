// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
lvis explores how a hierarchy of areas in the ventral stream of visual
processing (up to inferotemporal (IT) cortex) can produce robust object
recognition that is invariant to changes in position, size, etc of retinal
input images.
*/
package main

import (
	"log"
	"os"

	"github.com/emer/axon/axon"
	"github.com/emer/emergent/decoder"
	"github.com/emer/emergent/ecmd"
	"github.com/emer/emergent/egui"
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/estats"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/looper"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/empi/empi"
	"github.com/emer/empi/mpi"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/etview"
	_ "github.com/emer/etable/etview" // include to get gui views
	"github.com/emer/etable/minmax"
	"github.com/emer/etable/split"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
)

// Debug triggers various messages etc
var Debug = false

func main() {
	TheSim.New()
	if len(os.Args) > 1 {
		TheSim.CmdArgs() // simple assumption is that any args = no gui -- could add explicit arg if you want
	} else {
		gimain.Main(func() { // this starts gui -- requires valid OpenGL display connection (e.g., X11)
			guirun()
		})
	}
}

func guirun() {
	TheSim.Config()
	TheSim.Init()
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}

// see params.go for params

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Net          *axon.Network    `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	Prjns        Prjns            `desc:"special projections"`
	Params       emer.Params      `view:"inline" desc:"all parameter management"`
	Loops        *looper.Manager  `view:"no-inline" desc:"contains looper control loops for running sim"`
	Stats        estats.Stats     `desc:"contains computed statistic values"`
	Logs         elog.Logs        `desc:"Contains all the logs and information about the logs.'"`
	Envs         env.Envs         `view:"no-inline" desc:"Environments"`
	Time         axon.Time        `desc:"axon timing parameters and state"`
	ViewUpdt     netview.ViewUpdt `view:"inline" desc:"netview update parameters"`
	TestInterval int              `desc:"how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"`
	PCAInterval  int              `desc:"how frequently (in epochs) to compute PCA on hidden representations to measure variance?"`
	MaxTrls      int              `desc:"maximum number of training trials per epoch"`
	Decoder      decoder.SoftMax  `desc:"decoder for better output"`
	NOutPer      int              `desc:"number of units per localist output unit"`
	SubPools     bool             `desc:"if true, organize layers and connectivity with 2x2 sub-pools within each topological pool"`
	RndOutPats   bool             `desc:"if true, use random output patterns -- else localist"`
	ConfusionEpc int              `desc:"epoch to start recording confusion matrix"`

	GUI      egui.GUI    `view:"-" desc:"manages all the gui elements"`
	Args     ecmd.Args   `view:"no-inline" desc:"command line args"`
	RndSeeds erand.Seeds `view:"-" desc:"a list of random seeds to use for each run"`
	Comm     *mpi.Comm   `view:"-" desc:"mpi communicator"`
	AllDWts  []float32   `view:"-" desc:"buffer of all dwt weight changes -- for mpi sharing"`
	SumDWts  []float32   `view:"-" desc:"buffer of MPI summed dwt weight changes"`
}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for args when calling methods
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &axon.Network{}
	ss.Prjns.New()
	ss.Params.Params = ParamSets
	ss.Params.AddNetwork(ss.Net)
	ss.Params.AddSim(ss)
	ss.Params.AddNetSize()
	ss.Stats.Init()
	ss.RndSeeds.Init(100) // max 100 runs
	ss.NOutPer = 5
	ss.SubPools = true
	ss.RndOutPats = false
	ss.TestInterval = 20
	ss.PCAInterval = 10
	ss.ConfusionEpc = 500
	ss.MaxTrls = 512
	ss.Time.Defaults()
	ss.ConfigArgs() // do this first, has key defaults
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigLogs()
	ss.ConfigLoops()
}

func (ss *Sim) ConfigEnv() {
	// Can be called multiple times -- don't re-create
	var trn, tst *ImagesEnv
	if len(ss.Envs) == 0 {
		trn = &ImagesEnv{}
		tst = &ImagesEnv{}
	} else {
		trn = ss.Envs.ByMode(etime.Train).(*ImagesEnv)
		tst = ss.Envs.ByMode(etime.Test).(*ImagesEnv)
	}

	plus := false // plus images are a bit worse overall -- stranger objects etc.

	var path string
	if plus {
		path = "images/CU3D_100_plus_renders"
		trn.ImageFile = "cu3d100plus"
	} else {
		path = "images/CU3D_100_renders_lr20_u30_nb"
		trn.ImageFile = "cu3d100old"
	}

	trn.Nm = etime.Train.String()
	trn.Dsc = "training params and state"
	trn.Defaults()
	trn.High16 = false // not useful -- may need more tuning?
	trn.ColorDoG = true
	trn.Images.NTestPerCat = 2
	trn.Images.SplitByItm = true
	trn.OutRandom = ss.RndOutPats
	trn.OutSize.Set(10, 10)
	trn.Images.SetPath(path, []string{".png"}, "_")
	trn.OpenConfig()
	// trn.Images.OpenPath(path, []string{".png"}, "_")
	// trn.SaveConfig()

	trn.Validate()
	trn.Trial.Max = ss.MaxTrls

	tst.Nm = etime.Test.String()
	tst.Dsc = "testing params and state"
	tst.ImageFile = trn.ImageFile
	tst.Defaults()
	tst.High16 = trn.High16
	tst.ColorDoG = trn.ColorDoG
	tst.Images.NTestPerCat = 2
	tst.Images.SplitByItm = true
	tst.OutRandom = ss.RndOutPats
	tst.OutSize.Set(10, 10)
	tst.Test = true
	tst.Images.SetPath(path, []string{".png"}, "_")
	tst.OpenConfig()
	// tst.Images.OpenPath(path, []string{".png"}, "_")
	// tst.SaveConfig()
	tst.Trial.Max = ss.MaxTrls
	tst.Validate()

	/*
		// Delete to 60
			last20 := []string{"submarine", "synthesizer", "tablelamp", "tank", "telephone", "television", "toaster", "toilet", "trafficcone", "trafficlight", "trex", "trombone", "tropicaltree", "trumpet", "turntable", "umbrella", "wallclock", "warningsign", "wrench", "yacht"}
			next20 := []string{"pedestalsink", "person", "piano", "plant", "plate", "pliers", "propellor", "remote", "rolltopdesk", "sailboat", "scissors", "screwdriver", "sectionalcouch", "simpledesk", "skateboard", "skull", "slrcamera", "speaker", "spotlightlamp", "stapler"}
			last40 := append(last20, next20...)
			trn.Images.DeleteCats(last40)
			tst.Images.DeleteCats(last40)
	*/

	/*
		objs20 := []string{"banana", "layercake", "trafficcone", "sailboat", "trex", "person", "guitar", "tablelamp", "doorknob", "handgun", "donut", "chair", "slrcamera", "elephant", "piano", "fish", "car", "heavycannon", "stapler", "motorcycle"}

		objsnxt20 := []string{"submarine", "synthesizer", "tank", "telephone", "television", "toaster", "toilet", "trafficlight", "tropicaltree", "trumpet", "turntable", "umbrella", "wallclock", "warningsign", "wrench", "yacht", "pedestalsink", "pliers", "sectionalcouch", "skull"}

		objs40 := append(objs20, objsnxt20...)

		trn.Images.SelectCats(objs40)
		tst.Images.SelectCats(objs40)
	*/

	// remove most confusable items
	confuse := []string{"blade", "flashlight", "pckeyboard", "scissors", "screwdriver", "submarine"}
	trn.Images.DeleteCats(confuse)
	tst.Images.DeleteCats(confuse)

	if ss.Args.Bool("mpi") {
		if Debug {
			mpi.Printf("Did Env MPIAlloc\n")
		}
		trn.MPIAlloc()
		tst.MPIAlloc()
	}

	trn.Init(0)
	tst.Init(0)

	ss.Envs.Add(trn, tst)
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.InitName(net, "Lvis")

	trn := ss.Envs.ByMode(etime.Train).(*ImagesEnv)

	v1nrows := 5
	if trn.V1m16.SepColor {
		v1nrows += 4
	}
	hi16 := trn.High16
	cdog := trn.ColorDoG

	v2mNp := 8
	v2lNp := 4
	v2Nu := 8
	v4Np := 4
	v4Nu := 10
	if ss.SubPools {
		v2mNp *= 2
		v2lNp *= 2
		v2Nu = 6
		v4Np = 8
		v4Nu = 7
	}

	v1m16 := net.AddLayer4D("V1m16", 16, 16, v1nrows, 4, emer.Input)
	v1l16 := net.AddLayer4D("V1l16", 8, 8, v1nrows, 4, emer.Input)
	v1m8 := net.AddLayer4D("V1m8", 16, 16, v1nrows, 4, emer.Input)
	v1l8 := net.AddLayer4D("V1l8", 8, 8, v1nrows, 4, emer.Input)
	v1m16.SetClass("V1m")
	v1l16.SetClass("V1l")
	v1m8.SetClass("V1m")
	v1l8.SetClass("V1l")

	v1m16.SetRepIdxsShape(ss.CenterPoolIdxs(v1m16, 2), emer.CenterPoolShape(v1m16, 2))
	v1l16.SetRepIdxsShape(ss.CenterPoolIdxs(v1l16, 2), emer.CenterPoolShape(v1l16, 2))
	v1m8.SetRepIdxsShape(ss.CenterPoolIdxs(v1m8, 2), emer.CenterPoolShape(v1m8, 2))
	v1l8.SetRepIdxsShape(ss.CenterPoolIdxs(v1l8, 2), emer.CenterPoolShape(v1l8, 2))

	// not useful so far..
	// clst := net.AddLayer2D("Claustrum", 5, 5, emer.Hidden)

	var v1cm16, v1cl16, v1cm8, v1cl8 emer.Layer
	if cdog {
		v1cm16 = net.AddLayer4D("V1Cm16", 16, 16, 2, 2, emer.Input)
		v1cl16 = net.AddLayer4D("V1Cl16", 8, 8, 2, 2, emer.Input)
		v1cm8 = net.AddLayer4D("V1Cm8", 16, 16, 2, 2, emer.Input)
		v1cl8 = net.AddLayer4D("V1Cl8", 8, 8, 2, 2, emer.Input)
		v1cm16.SetClass("V1Cm")
		v1cl16.SetClass("V1Cl")
		v1cm8.SetClass("V1Cm")
		v1cl8.SetClass("V1Cl")

		v1cm16.SetRepIdxsShape(ss.CenterPoolIdxs(v1cm16, 2), emer.CenterPoolShape(v1cm16, 2))
		v1cl16.SetRepIdxsShape(ss.CenterPoolIdxs(v1cl16, 2), emer.CenterPoolShape(v1cl16, 2))
		v1cm8.SetRepIdxsShape(ss.CenterPoolIdxs(v1cm8, 2), emer.CenterPoolShape(v1cm8, 2))
		v1cl8.SetRepIdxsShape(ss.CenterPoolIdxs(v1cl8, 2), emer.CenterPoolShape(v1cl8, 2))
	}

	v2m16 := net.AddLayer4D("V2m16", v2mNp, v2mNp, v2Nu, v2Nu, emer.Hidden)
	v2l16 := net.AddLayer4D("V2l16", v2lNp, v2lNp, v2Nu, v2Nu, emer.Hidden)
	v2m8 := net.AddLayer4D("V2m8", v2mNp, v2mNp, v2Nu, v2Nu, emer.Hidden)
	v2l8 := net.AddLayer4D("V2l8", v2lNp, v2lNp, v2Nu, v2Nu, emer.Hidden)
	v2m16.SetClass("V2m V2")
	v2l16.SetClass("V2l V2")
	v2m8.SetClass("V2m V2")
	v2l8.SetClass("V2l V2")

	v2m16.SetRepIdxsShape(ss.CenterPoolIdxs(v2m16, 2), emer.CenterPoolShape(v2m16, 2))
	v2l16.SetRepIdxsShape(ss.CenterPoolIdxs(v2l16, 2), emer.CenterPoolShape(v2l16, 2))
	v2m8.SetRepIdxsShape(ss.CenterPoolIdxs(v2m8, 2), emer.CenterPoolShape(v2m8, 2))
	v2l8.SetRepIdxsShape(ss.CenterPoolIdxs(v2l8, 2), emer.CenterPoolShape(v2l8, 2))

	var v1h16, v2h16, v3h16 emer.Layer
	if hi16 {
		v1h16 = net.AddLayer4D("V1h16", 32, 32, 5, 4, emer.Input)
		v2h16 = net.AddLayer4D("V2h16", 32, 32, v2Nu, v2Nu, emer.Hidden)
		v3h16 = net.AddLayer4D("V3h16", 16, 16, v2Nu, v2Nu, emer.Hidden)
		v1h16.SetClass("V1h")
		v2h16.SetClass("V2h V2")
		v3h16.SetClass("V3h")

		v1h16.SetRepIdxsShape(ss.CenterPoolIdxs(v1h16, 2), emer.CenterPoolShape(v1h16, 2))
		v2h16.SetRepIdxsShape(ss.CenterPoolIdxs(v2h16, 2), emer.CenterPoolShape(v2h16, 2))
		v3h16.SetRepIdxsShape(ss.CenterPoolIdxs(v3h16, 2), emer.CenterPoolShape(v3h16, 2))
	}

	v4f16 := net.AddLayer4D("V4f16", v4Np, v4Np, v4Nu, v4Nu, emer.Hidden)
	v4f8 := net.AddLayer4D("V4f8", v4Np, v4Np, v4Nu, v4Nu, emer.Hidden)
	v4f16.SetClass("V4")
	v4f8.SetClass("V4")

	v4f16.SetRepIdxsShape(ss.CenterPoolIdxs(v4f16, 2), emer.CenterPoolShape(v4f16, 2))
	v4f8.SetRepIdxsShape(ss.CenterPoolIdxs(v4f8, 2), emer.CenterPoolShape(v4f8, 2))

	teo16 := net.AddLayer4D("TEOf16", 2, 2, 15, 15, emer.Hidden)
	teo8 := net.AddLayer4D("TEOf8", 2, 2, 15, 15, emer.Hidden)
	teo16.SetClass("TEO")
	teo8.SetClass("TEO")

	te := net.AddLayer4D("TE", 2, 2, 15, 15, emer.Hidden)

	var out emer.Layer
	if ss.RndOutPats {
		out = net.AddLayer2D("Output", trn.OutSize.Y, trn.OutSize.X, emer.Target)
	} else {
		out = net.AddLayer4D("Output", trn.OutSize.Y, trn.OutSize.X, trn.NOutPer, 1, emer.Target)
	}

	full := prjn.NewFull()
	_ = full
	rndcut := prjn.NewUnifRnd()
	rndcut.PCon = 0.1 // 0.2 == .1 459
	// rndprjn := prjn.NewUnifRnd()
	// rndprjn.PCon = 0.5 // 0.2 > .1
	pool1to1 := prjn.NewPoolOneToOne()
	_ = pool1to1

	pj := &ss.Prjns

	var p4x4s2, p2x2s1, p4x4s2send, p2x2s1send, p4x4s2recip, p2x2s1recip, v4toteo, teotov4 prjn.Pattern
	p4x4s2 = pj.Prjn4x4Skp2
	p2x2s1 = pj.Prjn2x2Skp1
	p4x4s2send = pj.Prjn4x4Skp2
	p2x2s1send = pj.Prjn2x2Skp1
	p4x4s2recip = pj.Prjn4x4Skp2Recip
	p2x2s1recip = pj.Prjn2x2Skp1Recip
	v4toteo = full
	teotov4 = full

	if ss.SubPools {
		p4x4s2 = pj.Prjn4x4Skp2Sub2
		p2x2s1 = pj.Prjn2x2Skp1Sub2
		p4x4s2send = pj.Prjn4x4Skp2Sub2Send
		p2x2s1send = pj.Prjn2x2Skp1Sub2Send
		p4x4s2recip = pj.Prjn4x4Skp2Sub2SendRecip
		p2x2s1recip = pj.Prjn2x2Skp1Sub2SendRecip
		v4toteo = pj.Prjn4x4Skp0Sub2
		teotov4 = pj.Prjn4x4Skp0Sub2Recip
	}

	net.ConnectLayers(v1m16, v2m16, p4x4s2, emer.Forward).SetClass("V1V2")
	net.ConnectLayers(v1l16, v2m16, p2x2s1, emer.Forward).SetClass("V1V2fmSm V1V2")

	net.ConnectLayers(v1l16, v2l16, p4x4s2, emer.Forward).SetClass("V1V2")

	net.ConnectLayers(v1m8, v2m8, p4x4s2, emer.Forward).SetClass("V1V2")
	net.ConnectLayers(v1l8, v2m8, p2x2s1, emer.Forward).SetClass("V1V2fmSm V1V2")

	net.ConnectLayers(v1l8, v2l8, p4x4s2, emer.Forward).SetClass("V1V2")

	if cdog {
		net.ConnectLayers(v1cm16, v2m16, p4x4s2, emer.Forward).SetClass("V1V2")
		net.ConnectLayers(v1cl16, v2m16, p2x2s1, emer.Forward).SetClass("V1V2fmSm V1V2")

		net.ConnectLayers(v1cl16, v2l16, p4x4s2, emer.Forward).SetClass("V1V2")

		net.ConnectLayers(v1cm8, v2m8, p4x4s2, emer.Forward).SetClass("V1V2")
		net.ConnectLayers(v1cl8, v2m8, p2x2s1, emer.Forward).SetClass("V1V2fmSm V1V2")

		net.ConnectLayers(v1cl8, v2l8, p4x4s2, emer.Forward).SetClass("V1V2")
	}

	v2v4, v4v2 := net.BidirConnectLayers(v2m16, v4f16, p4x4s2send)
	v2v4.SetClass("V2V4")
	v4v2.SetClass("V4V2").SetPattern(p4x4s2recip)

	v2v4, v4v2 = net.BidirConnectLayers(v2l16, v4f16, p2x2s1send)
	v2v4.SetClass("V2V4sm")
	v4v2.SetClass("V4V2").SetPattern(p2x2s1recip)

	v2v4, v4v2 = net.BidirConnectLayers(v2m8, v4f8, p4x4s2send)
	v2v4.SetClass("V2V4")
	v4v2.SetClass("V4V2").SetPattern(p4x4s2recip)

	v2v4, v4v2 = net.BidirConnectLayers(v2l8, v4f8, p2x2s1send)
	v2v4.SetClass("V2V4sm")
	v4v2.SetClass("V4V2").SetPattern(p2x2s1recip)

	if hi16 {
		net.ConnectLayers(v1h16, v2h16, p4x4s2, emer.Forward).SetClass("V1V2")
		v2v3, v3v2 := net.BidirConnectLayers(v2h16, v3h16, p4x4s2send)
		v2v3.SetClass("V2V3")
		v3v2.SetClass("V3V2").SetPattern(p4x4s2recip)
		v3v4, v4v3 := net.BidirConnectLayers(v3h16, v4f16, p4x4s2send)
		v3v4.SetClass("V3V4")
		v4v3.SetClass("V4V3").SetPattern(p4x4s2recip)
	}

	v4teo, teov4 := net.BidirConnectLayers(v4f16, teo16, v4toteo)
	v4teo.SetClass("V4TEO")
	teov4.SetClass("TEOV4").SetPattern(teotov4)
	net.ConnectLayers(v4f8, teo16, v4toteo, emer.Forward).SetClass("V4TEOoth")

	v4teo, teov4 = net.BidirConnectLayers(v4f8, teo8, v4toteo)
	v4teo.SetClass("V4TEO")
	teov4.SetClass("TEOV4").SetPattern(teotov4)
	net.ConnectLayers(v4f16, teo8, v4toteo, emer.Forward).SetClass("V4TEOoth")

	teote, teteo := net.BidirConnectLayers(teo16, te, full)
	teote.SetClass("TEOTE")
	teteo.SetClass("TETEO")
	teote, teteo = net.BidirConnectLayers(teo8, te, full)
	teote.SetClass("TEOTE")
	teteo.SetClass("TETEO")

	// full connections to output are key
	teoout, outteo := net.BidirConnectLayers(teo16, out, full)
	teoout.SetClass("TEOOut ToOut")
	outteo.SetClass("OutTEO FmOut")

	teoout, outteo = net.BidirConnectLayers(teo8, out, full)
	teoout.SetClass("TEOOut ToOut")
	outteo.SetClass("OutTEO FmOut")

	teout, _ := net.BidirConnectLayers(te, out, full)
	teout.SetClass("ToOut FmOut")

	// v59 459 -- only useful later -- TEO maybe not doing as well later?
	v4out, outv4 := net.BidirConnectLayers(v4f16, out, full)
	v4out.SetClass("V4Out ToOut")
	outv4.SetClass("OutV4 FmOut")

	v4out, outv4 = net.BidirConnectLayers(v4f8, out, full)
	v4out.SetClass("V4Out ToOut")
	outv4.SetClass("OutV4 FmOut")

	var v2inhib, v4inhib prjn.Pattern
	v2inhib = pool1to1
	v4inhib = pool1to1
	if ss.SubPools {
		v2inhib = pj.Prjn2x2Skp2 // pj.Prjn6x6Skp2Lat
		v4inhib = pj.Prjn2x2Skp2
	}

	// this extra inhibition drives decorrelation, produces significant learning benefits
	net.LateralConnectLayerPrjn(v2m16, v2inhib, &axon.HebbPrjn{}).SetType(emer.Inhib)
	net.LateralConnectLayerPrjn(v2l16, v2inhib, &axon.HebbPrjn{}).SetType(emer.Inhib)
	net.LateralConnectLayerPrjn(v2m8, v2inhib, &axon.HebbPrjn{}).SetType(emer.Inhib)
	net.LateralConnectLayerPrjn(v2l8, v2inhib, &axon.HebbPrjn{}).SetType(emer.Inhib)
	net.LateralConnectLayerPrjn(v4f16, v4inhib, &axon.HebbPrjn{}).SetType(emer.Inhib)
	net.LateralConnectLayerPrjn(v4f8, v4inhib, &axon.HebbPrjn{}).SetType(emer.Inhib)
	net.LateralConnectLayerPrjn(teo16, pool1to1, &axon.HebbPrjn{}).SetType(emer.Inhib)
	net.LateralConnectLayerPrjn(teo8, pool1to1, &axon.HebbPrjn{}).SetType(emer.Inhib)
	net.LateralConnectLayerPrjn(te, pool1to1, &axon.HebbPrjn{}).SetType(emer.Inhib)

	if hi16 {
		net.LateralConnectLayerPrjn(v2h16, v2inhib, &axon.HebbPrjn{}).SetType(emer.Inhib)
		net.LateralConnectLayerPrjn(v3h16, v2inhib, &axon.HebbPrjn{}).SetType(emer.Inhib)
	}

	///////////////////////
	// 	Shortcuts:

	// clst not useful
	// net.ConnectLayers(v1l16, clst, full, emer.Forward)

	// V1 shortcuts best for syncing all layers -- like the pulvinar basically
	net.ConnectLayers(v1l16, v4f16, rndcut, emer.Forward).SetClass("V1SC")
	net.ConnectLayers(v1l8, v4f8, rndcut, emer.Forward).SetClass("V1SC")
	net.ConnectLayers(v1l16, teo16, rndcut, emer.Forward).SetClass("V1SC")
	net.ConnectLayers(v1l16, teo16, rndcut, emer.Forward).SetClass("V1SC")
	net.ConnectLayers(v1l8, teo8, rndcut, emer.Forward).SetClass("V1SC")
	net.ConnectLayers(v1l8, teo8, rndcut, emer.Forward).SetClass("V1SC")
	net.ConnectLayers(v1l16, te, rndcut, emer.Forward).SetClass("V1SC")
	net.ConnectLayers(v1l8, te, rndcut, emer.Forward).SetClass("V1SC")

	if hi16 {
		net.ConnectLayers(v1l16, v3h16, rndcut, emer.Forward).SetClass("V1SC")
	}

	//////////////////////
	// 	Positioning

	v1m8.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: v1m16.Name(), YAlign: relpos.Front, Space: 4})

	v1l16.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: v1m16.Name(), XAlign: relpos.Left, Space: 4})
	v1l8.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: v1m8.Name(), XAlign: relpos.Left, Space: 4})
	// clst.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: v1l8.Name(), XAlign: relpos.Left, Space: 4, Scale: 2})

	if cdog {
		v1cm16.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: v1m8.Name(), YAlign: relpos.Front, Space: 4})
		v1cm8.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: v1cm16.Name(), YAlign: relpos.Front, Space: 4})
		v1cl16.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: v1cm16.Name(), XAlign: relpos.Left, Space: 4})
		v1cl8.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: v1cm8.Name(), XAlign: relpos.Left, Space: 4})
	}

	if hi16 {
		v1h16.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: v1m8.Name(), YAlign: relpos.Front, Space: 4})
		v2h16.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: v2m8.Name(), YAlign: relpos.Front, Space: 4})
		v3h16.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: v4f16.Name(), XAlign: relpos.Left, Space: 4})
	}

	v2m16.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: v1m16.Name(), XAlign: relpos.Left, YAlign: relpos.Front})

	v2m8.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: v2m16.Name(), YAlign: relpos.Front, Space: 4})

	v2l16.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: v2m16.Name(), XAlign: relpos.Left, Space: 4})
	v2l8.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: v2m8.Name(), XAlign: relpos.Left, Space: 4})

	v4f16.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: v2m16.Name(), XAlign: relpos.Left, YAlign: relpos.Front})
	teo16.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: v4f16.Name(), YAlign: relpos.Front, Space: 4})

	v4f8.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: teo16.Name(), XAlign: relpos.Left, YAlign: relpos.Front})
	teo8.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: v4f8.Name(), YAlign: relpos.Front, Space: 4})

	te.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: teo8.Name(), XAlign: relpos.Left, Space: 15})

	out.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: te.Name(), XAlign: relpos.Left, Space: 15})

	if hi16 {
		v3h16.SetThread(1)
	}

	v4f16.SetThread(1)
	v4f8.SetThread(1)

	teo16.SetThread(1)
	teo8.SetThread(1)
	te.SetThread(1)
	out.SetThread(1)

	net.Defaults()
	ss.Params.SetObject("Network")
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	ss.Net.InitWts()

	if !ss.Args.Bool("nogui") {
		sr := net.SizeReport()
		mpi.Printf("%s", sr)
	}
	ar := net.ThreadReport() // hand tuning now..
	mpi.Printf("%s", ar)

	// adding each additional layer type improves decoding..
	layers := []emer.Layer{v4f16, v4f8, teo16, teo8, out}
	// layers := []emer.Layer{teo16, teo8, out}
	// layers := []emer.Layer{teo16, teo8}
	ss.Decoder.InitLayer(len(trn.Images.Cats), layers)
	ss.Decoder.Lrate = 0.05 // 0.05 > 0.1 > 0.2 for larger number of objs!
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.Loops.ResetCounters()
	ss.InitRndSeed()
	// ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.GUI.StopNow = false
	ss.Params.SetAll()
	ss.Net.SlowInterval = 100 // 100 > 20
	ss.NewRun()
	ss.ViewUpdt.Update()
}

// InitRndSeed initializes the random seed based on current training run number
func (ss *Sim) InitRndSeed() {
	run := ss.Loops.GetLoop(etime.Train, etime.Run).Counter.Cur
	ss.RndSeeds.Set(run)
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	man := looper.NewManager()

	effTrls := ss.MaxTrls
	if ss.Args.Bool("mpi") {
		effTrls /= mpi.WorldSize() // todo: use more robust fun
		if Debug {
			mpi.Printf("MPI trials: %d\n", effTrls)
		}
	}

	man.AddStack(etime.Train).AddTime(etime.Run, 1).AddTime(etime.Epoch, 2000).AddTime(etime.Trial, effTrls).AddTime(etime.Cycle, 200)

	man.AddStack(etime.Test).AddTime(etime.Epoch, 1).AddTime(etime.Trial, effTrls).AddTime(etime.Cycle, 200)

	axon.LooperStdPhases(man, &ss.Time, ss.Net.AsAxon(), 150, 199)            // plus phase timing
	axon.LooperSimCycleAndLearn(man, ss.Net.AsAxon(), &ss.Time, &ss.ViewUpdt) // std algo code

	man.GetLoop(etime.Train, etime.Trial).OnEnd.Replace("UpdateWeights", func() {
		ss.Net.DWt(&ss.Time)
		ss.ViewUpdt.RecordSyns() // note: critical to update weights here so DWt is visible
		ss.MPIWtFmDWt()
	})

	for m, _ := range man.Stacks {
		mode := m // For closures
		stack := man.Stacks[mode]
		stack.Loops[etime.Trial].OnStart.Add("Env:Step", func() {
			// note: OnStart for env.Env, others may happen OnEnd
			ss.Envs[mode.String()].Step()
		})
		stack.Loops[etime.Trial].OnStart.Add("ApplyInputs", func() {
			ss.ApplyInputs()
			// axon.EnvApplyInputs(ss.Net, ss.Envs[ss.Time.Mode])
		})
		stack.Loops[etime.Trial].OnEnd.Add("StatCounters", ss.StatCounters)
		stack.Loops[etime.Trial].OnEnd.Add("TrialStats", ss.TrialStats)
	}

	man.GetLoop(etime.Train, etime.Run).OnStart.Add("NewRun", ss.NewRun)

	// Train stop early condition
	man.GetLoop(etime.Train, etime.Epoch).IsDone["NZeroStop"] = func() bool {
		// This is calculated in TrialStats
		stopNz := ss.Args.Int("nzero")
		if stopNz <= 0 {
			stopNz = 2
		}
		curNZero := ss.Stats.Int("NZero")
		stop := curNZero >= stopNz
		return stop
	}

	// Add Testing
	trainEpoch := man.GetLoop(etime.Train, etime.Epoch)
	trainEpoch.OnStart.Add("TestAtInterval", func() {
		if (ss.TestInterval > 0) && ((trainEpoch.Counter.Cur+1)%ss.TestInterval == 0) {
			// Note the +1 so that it doesn't occur at the 0th timestep.
			ss.TestAll()
		}
	})

	trainEpoch.OnEnd.Add("RandCheck", func() {
		if ss.Args.Bool("mpi") {
			empi.RandCheck(ss.Comm) // prints error message
		}
	})

	/////////////////////////////////////////////
	// Logging

	man.GetLoop(etime.Test, etime.Epoch).OnEnd.Add("LogTestErrors", func() {
		axon.LogTestErrors(&ss.Logs)
	})
	man.GetLoop(etime.Train, etime.Epoch).OnEnd.Add("PCAStats", func() {
		trnEpc := man.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
		if (ss.PCAInterval > 0) && (trnEpc%ss.PCAInterval == 0) {
			if ss.Args.Bool("mpi") {
				ss.Logs.MPIGatherTableRows(etime.Analyze, etime.Trial, ss.Comm)
			}
			axon.PCAStats(ss.Net.AsAxon(), &ss.Logs, &ss.Stats)
			ss.Logs.ResetLog(etime.Analyze, etime.Trial)
		}
	})

	man.AddOnEndToAll("Log", ss.Log)
	axon.LooperResetLogBelow(man, &ss.Logs)

	man.GetLoop(etime.Train, etime.Trial).OnEnd.Add("LogAnalyze", func() {
		trnEpc := man.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
		if (ss.PCAInterval > 0) && (trnEpc%ss.PCAInterval == 0) {
			ss.Log(etime.Analyze, etime.Trial)
		}
	})

	man.GetLoop(etime.Train, etime.Run).OnEnd.Add("RunStats", func() {
		ss.Logs.RunStats("PctCor", "FirstZero", "LastZero")
	})

	// Save weights to file at end, to look at later
	man.GetLoop(etime.Train, etime.Run).OnEnd.Add("SaveWeights", func() { ss.SaveWeights() })

	man.GetLoop(etime.Train, etime.Epoch).AddNewEvent("SaveWeights", 100, func() { ss.SaveWeights() })
	man.GetLoop(etime.Train, etime.Epoch).AddNewEvent("SaveWeights", 500, func() { ss.SaveWeights() })
	man.GetLoop(etime.Train, etime.Epoch).AddNewEvent("SaveWeights", 1000, func() { ss.SaveWeights() })
	man.GetLoop(etime.Train, etime.Epoch).AddNewEvent("SaveWeights", 1500, func() { ss.SaveWeights() })

	man.GetLoop(etime.Test, etime.Trial).OnEnd.Add("ActRFs", func() {
		ss.Stats.UpdateActRFs(ss.Net, "ActM", 0.01)
	})

	////////////////////////////////////////////
	// GUI
	if ss.Args.Bool("nogui") {
		man.GetLoop(etime.Test, etime.Trial).Main.Add("NetDataRecord", func() {
			ss.GUI.NetDataRecord(ss.ViewUpdt.Text)
		})
	} else {
		axon.LooperUpdtNetView(man, &ss.ViewUpdt)
		axon.LooperUpdtPlots(man, &ss.GUI)
	}

	if Debug {
		mpi.Println(man.DocString())
	}
	ss.Loops = man
}

// SaveWeights saves weights with filename recording run, epoch
func (ss *Sim) SaveWeights() {
	ctrString := ss.Stats.PrintVals([]string{"Run", "Epoch"}, []string{"%03d", "%05d"}, "_")
	axon.SaveWeightsIfArgSet(ss.Net.AsAxon(), &ss.Args, ctrString, ss.Stats.String("RunName"))
}

// CenterPoolIdxs returns the unit indexes for 2x2 center pools
// if sub-pools are present, then only first such subpool is used.
func (ss *Sim) CenterPoolIdxs(ly emer.Layer, n int) []int {
	npy := ly.Shape().Dim(0)
	npx := ly.Shape().Dim(1)
	npxact := npx
	nu := ly.Shape().Dim(2) * ly.Shape().Dim(3)
	nsp := 1
	if ss.SubPools {
		npy /= 2
		npx /= 2
		nsp = 2
	}
	cpy := (npy - n) / 2
	cpx := (npx - n) / 2
	nt := n * n * nu
	idxs := make([]int, nt)

	ix := 0
	for py := 0; py < 2; py++ {
		y := (py + cpy) * nsp
		for px := 0; px < 2; px++ {
			x := (px + cpx) * nsp
			si := (y*npxact + x) * nu
			for ni := 0; ni < nu; ni++ {
				idxs[ix+ni] = si + ni
			}
			ix += nu
		}
	}
	return idxs
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs() {
	net := ss.Net
	ev := ss.Envs[ss.Time.Mode]
	net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway
	lays := net.LayersByClass("Input", "Target")
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		pats := ev.State(ly.Nm)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ss.InitRndSeed()
	ss.Envs.ByMode(etime.Train).Init(0)
	ss.Envs.ByMode(etime.Test).Init(0)
	ss.Time.Reset()
	ss.Time.Mode = etime.Train.String()
	ss.Net.InitWts()
	ss.InitStats()
	ss.StatCounters()
	ss.Logs.ResetLog(etime.Train, etime.Epoch)
	ss.Logs.ResetLog(etime.Test, etime.Epoch)
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.Envs.ByMode(etime.Test).Init(0)
	ss.Loops.Mode = etime.Test
	ss.Stats.ActRFs.Reset()
	ss.Loops.Run()
	ss.Loops.Mode = etime.Train // Important to reset Mode back to Train because this is called from within the Train Run.
	ss.Stats.ActRFsAvgNorm()
	ss.GUI.ViewActRFs(&ss.Stats.ActRFs)

}

// RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunTestAll() {
	ss.Logs.ResetLog(etime.Test, etime.Epoch) // only show last row
	ss.GUI.StopNow = false
	ss.TestAll()
	ss.GUI.Stopped()
}

// ConfusionTstPlot plots the current confusion probability values.
// if cat is empty then it is the diagonal accuracy across all cats
// otherwise it is the confusion row for given category.
// data goes in the TrlErr = Err column.
func (ss *Sim) ConfusionTstPlot(cat string) {
	env := ss.Envs[etime.Test.String()].(*ImagesEnv)
	ss.Logs.ResetLog(etime.Test, etime.Trial)
	nc := ss.Stats.Confusion.N.Len()
	ti := -1
	if cat != "" {
		ti = env.Images.CatMap[cat]
	}
	for i := 0; i < nc; i++ {
		env.Trial.Cur = i
		env.CurCat = env.Images.Cats[i]
		if ti >= 0 {
			ss.Stats.SetFloat("TrlErr", ss.Stats.Confusion.Prob.Value([]int{ti, i}))
		} else {
			ss.Stats.SetFloat("TrlErr", ss.Stats.Confusion.Prob.Value([]int{i, i}))
		}
		ss.Log(etime.Test, etime.Trial)
	}
	plt := ss.GUI.Plot(etime.Test, etime.Trial)
	plt.Params.XAxisCol = "Cat"
	plt.Params.Type = eplot.Bar
	plt.Params.XAxisRot = 45
	plt.GoUpdate()
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	ss.Stats.SetFloat("TrlUnitErr", 0.0)
	ss.Stats.SetFloat("TrlCorSim", 0.0)
	ss.Logs.InitErrStats() // inits TrlErr, FirstZero, LastZero, NZero
	ss.Stats.SetFloat("TrlErr2", 0.0)
	ss.Stats.SetString("TrlCat", "0")
	ss.Stats.SetInt("TrlCatIdx", 0)
	ss.Stats.SetInt("TrlRespIdx", 0)
	ss.Stats.SetInt("TrlDecRespIdx", 0)
	ss.Stats.SetFloat("TrlDecErr", 0.0)
	ss.Stats.SetFloat("TrlDecErr2", 0.0)
	env := ss.Envs[etime.Train.String()].(*ImagesEnv)
	ss.Stats.Confusion.InitFromLabels(env.Images.Cats, 12)
}

// StatCounters saves current counters to Stats, so they are available for logging etc
// Also saves a string rep of them for ViewUpdt.Text
func (ss *Sim) StatCounters() {
	var mode etime.Modes
	mode.FromString(ss.Time.Mode)
	ss.Loops.Stacks[mode].CtrsToStats(&ss.Stats)
	// always use training epoch..
	trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
	ss.Stats.SetInt("Epoch", trnEpc)
	ss.Stats.SetInt("Cycle", ss.Time.Cycle)
	ev := ss.Envs[ss.Time.Mode].(*ImagesEnv)
	ss.Stats.SetString("TrialName", ev.String())
	ss.Stats.SetString("TrlResp", "")
	ss.ViewUpdt.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "TrlCat", "TrlResp", "TrialName", "Cycle", "TrlUnitErr", "TrlErr", "TrlCorSim"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats() {
	out := ss.Net.LayerByName("Output").(axon.AxonLayer).AsAxon()

	ss.Stats.SetFloat("TrlCorSim", float64(out.CorSim.Cor))
	ss.Stats.SetFloat("TrlUnitErr", out.PctUnitErr())

	ovt := ss.Stats.SetLayerTensor(ss.Net, "Output", "ActM")
	env := ss.Envs[ss.Time.Mode].(*ImagesEnv)

	ncats := len(env.Images.Cats)

	rsp, trlErr, trlErr2 := env.OutErr(ovt)
	ss.Stats.SetInt("TrlRespIdx", rsp)
	ss.Stats.SetFloat("TrlErr", trlErr)
	ss.Stats.SetFloat("TrlErr2", trlErr2)
	if rsp >= 0 && rsp < ncats {
		ss.Stats.SetString("TrlResp", env.Images.Cats[rsp])
	}
	ss.Stats.SetInt("TrlCatIdx", env.CurCatIdx)
	ss.Stats.SetString("TrlCat", env.CurCat)

	epc := env.Epoch.Cur
	if epc > ss.ConfusionEpc {
		ss.Stats.Confusion.Incr(ss.Stats.Int("TrlCatIdx"), ss.Stats.Int("TrlRespIdx"))
	}

	ss.Stats.SetFloat("TrlTrgAct", float64(out.Pools[0].ActP.Avg/0.01))
	decIdx := ss.Decoder.Decode("ActM")
	ss.Stats.SetInt("TrlDecRespIdx", decIdx)
	if ss.Time.Mode == etime.Train.String() {
		ss.Decoder.Train(env.CurCatIdx)
	}
	decErr := float64(0)
	if decIdx != env.CurCatIdx {
		decErr = 1
	}
	ss.Stats.SetFloat("TrlDecErr", decErr)
	decErr2 := decErr
	if ss.Decoder.Sorted[1] == env.CurCatIdx {
		decErr2 = 0
	}
	ss.Stats.SetFloat("TrlDecErr2", decErr2)

	cyclog := ss.Logs.Log(etime.Train, etime.Cycle)
	var fcyc int
	fcyc, rsp, trlErr, trlErr2 = ss.FirstOut(cyclog)
	ss.Stats.SetInt("TrlOutFirstCyc", fcyc)
	ss.Stats.SetInt("TrlFirstResp", rsp)
	ss.Stats.SetFloat("TrlFirstErr", trlErr)
	ss.Stats.SetFloat("TrlFirstErr2", trlErr2)
}

// FindPeaks returns indexes of local maxima in input slice, smoothing
// the data first using GaussKernel
func (ss *Sim) FindPeaks(data []float64) []int {
	// convolve.Slice64(&ss.SmoothData, data, ss.GaussKernel)
	// dt := ss.SmoothData
	dt := data // already smooth
	sz := len(dt)
	off := 10
	peaks := []int{}
	for wd := 4; wd >= 1; wd-- {
		for i := off + wd; i < sz-wd; i++ {
			ctr := dt[i]
			fail := false
			for j := -wd; j <= wd; j++ {
				if dt[i+j] > ctr {
					fail = true
					break
				}
			}
			if !fail {
				peaks = append(peaks, i)
				i += wd
			}
		}
		if len(peaks) > 0 {
			break
		}
	}
	return peaks
}

// FindActCycle returns the point at which max activity stops going up by more than .01
// within minus phase.
// must be passed max act data cycle-by-cycle
func (ss *Sim) FindActCycle(data []float64) int {
	mx := 150
	if len(data) < mx {
		mpi.Printf("FindActCycle error: data is len: %d\n", len(data))
		return mx
	}
	dt := data  // data is already smooth
	start := 25 // give time for prior act to decay
	thr := 0.01 // rise threshold
	hit := false
	cyc := mx
	for i := start; i < mx; i++ {
		del := dt[i] - dt[i-1]
		if !hit {
			if del > thr {
				hit = true
			}
			continue
		}
		if del < thr {
			cyc = i
			break
		}
	}
	return cyc
}

// FirstActStat returns first major activation of given layer
func (ss *Sim) FirstActStat(cyclog *etable.Table, lnm string) int {
	dc := cyclog.ColByName(lnm + "_ActMax").(*etensor.Float64)
	return ss.FindActCycle(dc.Values)
}

// FirstOut returns first output response at first peak of output activity
func (ss *Sim) FirstOut(cyclog *etable.Table) (fcyc, resp int, err float64, err2 float64) {
	fcyc = ss.FirstActStat(cyclog, "Output")
	oval := cyclog.CellTensor("Output_Act", fcyc)
	if oval == nil {
		err = 1
		err2 = 1
		return
	}
	out := cyclog.CellTensor("Output_Act", fcyc).(*etensor.Float32)
	env := ss.Envs[ss.Time.Mode].(*ImagesEnv)
	resp, err, err2 = env.OutErr(out)
	return
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Trial, etime.Cycle)
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrlCat", "TrialName", "TrlResp")

	ss.Logs.AddStatAggItem("CorSim", "TrlCorSim", true, etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("UnitErr", "TrlUnitErr", false, etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddErrStatAggItems("TrlErr", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	ss.ConfigLogItems()
	ss.ConfigActRFs()

	axon.LogAddDiagnosticItems(&ss.Logs, ss.Net.AsAxon(), etime.Epoch, etime.Trial)
	axon.LogAddPCAItems(&ss.Logs, ss.Net.AsAxon(), etime.Run, etime.Epoch, etime.Trial)

	axon.LogAddLayerGeActAvgItems(&ss.Logs, ss.Net.AsAxon(), etime.Test, etime.Cycle)
	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Test, etime.Trial, "Target")
	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.AllModes, etime.Cycle, "Target")

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net.AsAxon())
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Run)
	// note: Analyze not plotted by default
	ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "RunName")
	ss.Logs.SetMeta(etime.Test, etime.Epoch, "Type", "Bar")
}

// ConfigLogItems specifies extra logging items
func (ss *Sim) ConfigLogItems() {
	ss.Logs.AddItem(&elog.Item{
		Name: "Err2",
		Type: etensor.FLOAT64,
		Plot: true,
		Write: elog.WriteMap{
			etime.Scope(etime.AllModes, etime.Trial): func(ctx *elog.Context) {
				ctx.SetStatFloat("TrlErr2")
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "PctErr2",
		Type: etensor.FLOAT64,
		Plot: false,
		Write: elog.WriteMap{
			etime.Scope(etime.AllModes, etime.Epoch): func(ctx *elog.Context) {
				ctx.SetAggItem(ctx.Mode, etime.Trial, "Err2", agg.AggMean)
			}, etime.Scope(etime.AllModes, etime.Run): func(ctx *elog.Context) {
				ix := ctx.LastNRows(ctx.Mode, etime.Epoch, 5)
				ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
			}}})
	// DecErr decoding
	ss.Logs.AddItem(&elog.Item{
		Name: "DecResp",
		Type: etensor.STRING,
		Plot: elog.DFalse,
		Write: elog.WriteMap{
			etime.Scope(etime.AllModes, etime.Trial): func(ctx *elog.Context) {
				ctx.SetStatInt("TrlDecRespIdx")
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "DecErr",
		Type: etensor.FLOAT64,
		Plot: elog.DTrue,
		Write: elog.WriteMap{
			etime.Scope(etime.AllModes, etime.Trial): func(ctx *elog.Context) {
				ctx.SetStatFloat("TrlDecErr")
			}, etime.Scope(etime.AllModes, etime.Epoch): func(ctx *elog.Context) {
				ctx.SetAgg(ctx.Mode, etime.Trial, agg.AggMean)
			}, etime.Scope(etime.AllModes, etime.Run): func(ctx *elog.Context) {
				ix := ctx.LastNRows(ctx.Mode, etime.Epoch, 5)
				ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "DecErr2",
		Type: etensor.FLOAT64,
		Plot: elog.DTrue,
		Write: elog.WriteMap{
			etime.Scope(etime.AllModes, etime.Trial): func(ctx *elog.Context) {
				ctx.SetStatFloat("TrlDecErr2")
			}, etime.Scope(etime.AllModes, etime.Epoch): func(ctx *elog.Context) {
				ctx.SetAgg(ctx.Mode, etime.Trial, agg.AggMean)
			}, etime.Scope(etime.AllModes, etime.Run): func(ctx *elog.Context) {
				ix := ctx.LastNRows(ctx.Mode, etime.Epoch, 5)
				ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
			}}})
	// FirstErr
	ss.Logs.AddItem(&elog.Item{
		Name: "FirstErr",
		Type: etensor.FLOAT64,
		Plot: elog.DTrue,
		Write: elog.WriteMap{
			etime.Scope(etime.AllModes, etime.Trial): func(ctx *elog.Context) {
				ctx.SetStatFloat("TrlFirstErr")
			}, etime.Scope(etime.AllModes, etime.Epoch): func(ctx *elog.Context) {
				ctx.SetAgg(ctx.Mode, etime.Trial, agg.AggMean)
			}, etime.Scope(etime.AllModes, etime.Run): func(ctx *elog.Context) {
				ix := ctx.LastNRows(ctx.Mode, etime.Epoch, 5)
				ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "FirstErr2",
		Type: etensor.FLOAT64,
		Plot: elog.DTrue,
		Write: elog.WriteMap{
			etime.Scope(etime.AllModes, etime.Trial): func(ctx *elog.Context) {
				ctx.SetStatFloat("TrlFirstErr2")
			}, etime.Scope(etime.AllModes, etime.Epoch): func(ctx *elog.Context) {
				ctx.SetAgg(ctx.Mode, etime.Trial, agg.AggMean)
			}, etime.Scope(etime.AllModes, etime.Run): func(ctx *elog.Context) {
				ix := ctx.LastNRows(ctx.Mode, etime.Epoch, 5)
				ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
			}}})

	ss.Logs.AddItem(&elog.Item{
		Name:      "CatErr",
		Type:      etensor.FLOAT64,
		CellShape: []int{100},
		DimNames:  []string{"Cat"},
		Plot:      true,
		Range:     minmax.F64{Min: 0},
		TensorIdx: -1, // plot all values
		Write: elog.WriteMap{
			etime.Scope(etime.Test, etime.Epoch): func(ctx *elog.Context) {
				ix := ctx.Logs.IdxView(etime.Test, etime.Trial)
				spl := split.GroupBy(ix, []string{"TrlCat"})
				split.AggTry(spl, "Err", agg.AggMean)
				cats := spl.AggsToTable(etable.ColNameOnly)
				ss.Logs.MiscTables[ctx.Item.Name] = cats
				ctx.SetTensor(cats.Cols[1])
			}}})

	layers := ss.Net.LayersByClass("Hidden", "Target")
	for _, lnm := range layers {
		clnm := lnm
		cly := ss.Net.LayerByName(clnm)
		ss.Logs.AddItem(&elog.Item{
			Name:   clnm + "_ActMax",
			Type:   etensor.FLOAT64,
			Plot:   elog.DFalse,
			FixMax: elog.DFalse,
			Range:  minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.AllModes, etime.Cycle): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].Inhib.Act.Max)
				}, etime.Scope(etime.AllModes, etime.Trial): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].Inhib.Act.Max)
				}, etime.Scope(etime.AllModes, etime.Epoch): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, etime.Trial, agg.AggMean)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:  clnm + "_FirstCyc",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Train, etime.Trial): func(ctx *elog.Context) {
					fcyc := ss.FirstActStat(ctx.Logs.Table(ctx.Mode, etime.Cycle), clnm)
					ctx.SetInt(fcyc)
				}, etime.Scope(etime.Train, etime.Epoch): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, etime.Trial, agg.AggMean)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:  clnm + "_FF_AvgMaxG",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.AllModes, etime.Trial): func(ctx *elog.Context) {
					ffpj := cly.RecvPrjn(0).(*axon.Prjn)
					ctx.SetFloat32(ffpj.GScale.AvgMax)
				}, etime.Scope(etime.AllModes, etime.Epoch): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, etime.Trial, agg.AggMean)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:  clnm + "_FB_AvgMaxG",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.AllModes, etime.Trial): func(ctx *elog.Context) {
					if cly.NRecvPrjns() > 1 {
						fbpj := cly.RecvPrjn(1).(*axon.Prjn)
						ctx.SetFloat32(fbpj.GScale.AvgMax)
					}
				}, etime.Scope(etime.AllModes, etime.Epoch): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, etime.Trial, agg.AggMean)
				}}})
		if clnm == "Output" {
			ss.Logs.AddItem(&elog.Item{
				Name:   clnm + "_GiMult",
				Type:   etensor.FLOAT64,
				Plot:   elog.DFalse,
				FixMax: elog.DFalse,
				Range:  minmax.F64{Max: 1},
				Write: elog.WriteMap{
					etime.Scope(etime.AllModes, etime.Epoch): func(ctx *elog.Context) {
						ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
						ctx.SetFloat32(ly.ActAvg.GiMult)
					}}})
		}
	}
}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode etime.Modes, time etime.Times) {
	if mode.String() != "Analyze" {
		ss.Time.Mode = mode.String() // Also set specifically in a Loop callback.
	}
	ss.StatCounters()

	if ss.Args.Bool("mpi") && time == etime.Epoch { // Must gather data for trial level if doing epoch level
		ss.Logs.MPIGatherTableRows(mode, etime.Trial, ss.Comm)
	}

	dt := ss.Logs.Table(mode, time)
	row := dt.Rows

	switch {
	case time == etime.Cycle:
		row = ss.Stats.Int("Cycle")
	case time == etime.Trial:
		row = ss.Stats.Int("Trial")
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc

	trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
	if trnEpc > ss.ConfusionEpc {
		ss.Stats.Confusion.Probs()
		fnm := ecmd.LogFileName("trn_conf", ss.Net.Name(), ss.Stats.String("RunName"))
		ss.Stats.Confusion.SaveCSV(gi.FileName(fnm))
	}
}

// ConfigActRFs
func (ss *Sim) ConfigActRFs() {
	ss.Stats.SetF32Tensor("Image", &ss.Envs[etime.Test.String()].(*ImagesEnv).Img.Tsr) // image used for actrfs, must be there first
	ss.Stats.InitActRFs(ss.Net, []string{"V4f16:Image", "V4f16:Output", "TEOf16:Image", "TEOf16:Output", "TEOf8:Image", "TEOf8:Output"}, "ActM")
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	nv.ViewDefaults()
	// cam := &(nv.Scene().Camera)
	// cam.Pose.Pos.Set(0.0, 1.733, 2.3)
	// cam.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})
	// cam.Pose.Quat.SetFromAxisAngle(mat32.Vec3{-1, 0, 0}, 0.4077744)
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	title := "LVis Object Recognition"
	ss.GUI.MakeWindow(ss, "lvis", title, `This simulation explores how a hierarchy of areas in the ventral stream of visual processing (up to inferotemporal (IT) cortex) can produce robust object recognition that is invariant to changes in position, size, etc of retinal input images. See <a href="https://github.com/ccnlab/lvis/blob/master/sims/lvis_cu3d100_te16deg_axon/README.md">README.md on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("NetView")
	nv.Params.MaxRecs = 300
	nv.SetNet(ss.Net)
	ss.ViewUpdt.Config(nv, etime.AlphaCycle, etime.AlphaCycle)
	ss.GUI.ViewUpdt = &ss.ViewUpdt
	ss.ConfigNetView(nv)

	ss.GUI.AddPlots(title, &ss.Logs)

	tg := ss.GUI.TabView.AddNewTab(etview.KiT_TensorGrid, "Image").(*etview.TensorGrid)
	tg.SetStretchMax()
	ss.GUI.SetGrid("Image", tg)
	tg.SetTensor(&ss.Envs[etime.Train.String()].(*ImagesEnv).Img.Tsr)

	ss.GUI.AddActRFGridTabs(&ss.Stats.ActRFs)

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Init", Icon: "update",
		Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Init()
			ss.GUI.UpdateWindow()
		},
	})

	ss.GUI.AddLooperCtrl(ss.Loops, []etime.Modes{etime.Train, etime.Test})

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Test All",
		Icon:    "step-fwd",
		Tooltip: "Tests a large same of testing items and records ActRFs.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				ss.GUI.IsRunning = true
				ss.GUI.ToolBar.UpdateActions()
				go ss.RunTestAll()
			}
		},
	})

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Conf To Test",
		Icon:    "fast-fwd",
		Tooltip: "Plots accuracy from current confusion probs to test trial log for each category (diagonal of confusion matrix).",
		Active:  egui.ActiveStopped,
		Func: func() {
			giv.CallMethod(ss, "ConfusionTstPlot", ss.GUI.ViewPort)
		},
	})

	////////////////////////////////////////////////
	ss.GUI.ToolBar.AddSeparator("log")
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Reset RunLog",
		Icon:    "reset",
		Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.Logs.ResetLog(etime.Train, etime.Run)
			ss.GUI.UpdatePlot(etime.Train, etime.Run)
		},
	})
	////////////////////////////////////////////////
	ss.GUI.ToolBar.AddSeparator("misc")
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "New Seed",
		Icon:    "new",
		Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time.",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.RndSeeds.NewSeeds()
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "README",
		Icon:    "file-markdown",
		Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
		Active:  egui.ActiveAlways,
		Func: func() {
			gi.OpenURL("https://github.com/lvis/blob/main/sims/lvis_cu3d100_te16deg_axon/README.md")
		},
	})
	ss.GUI.FinalizeGUI(false)
	return ss.GUI.Win
}

func (ss *Sim) ConfigArgs() {
	ss.Args.Init()
	ss.Args.AddStd()
	ss.Args.AddInt("nzero", 2, "number of zero error epochs in a row to count as full training")
	ss.Args.AddInt("iticycles", 0, "number of cycles to run between trials (inter-trial-interval)")
	ss.Args.SetInt("epochs", 2000)
	ss.Args.SetInt("runs", 1)
	ss.Args.AddBool("mpi", false, "if set, use MPI for distributed computation")
	ss.Args.Parse() // always parse
}

// These props register methods so they can be called through gui with arg prompts
var SimProps = ki.Props{
	"CallMethods": ki.PropSlice{
		{"ConfusionTstPlot", ki.Props{
			"desc": "plot current confusion matrix probs in TstTrlPlot -- enter Cat for confusion row for that category, else if blank, diagonal accuracy for all categories",
			"icon": "file-sheet",
			"Args": ki.PropSlice{
				{"Cat", ki.Props{
					"desc": "category name to show",
				}},
			},
		}},
	},
}

func (ss *Sim) CmdArgs() {
	if ss.Args.Bool("mpi") {
		ss.MPIInit()
	}

	// key for Config and Init to be after MPIInit
	ss.Config()
	ss.Init()

	ss.Args.ProcStd(&ss.Params)

	if mpi.WorldRank() == 0 {
		ss.Args.ProcStdLogs(&ss.Logs, &ss.Params, ss.Net.Name())
	}

	ss.Args.SetBool("nogui", true)                                       // by definition if here
	ss.Stats.SetString("RunName", ss.Params.RunName(ss.Args.Int("run"))) // used for naming logs, stats, etc

	if mpi.WorldRank() != 0 {
		ss.Args.SetBool("wts", false)
	}

	netdata := ss.Args.Bool("netdata")
	if netdata {
		mpi.Printf("Saving NetView data from testing\n")
		ss.GUI.InitNetData(ss.Net, 200)
	}

	runs := ss.Args.Int("runs")
	run := ss.Args.Int("run")
	mpi.Printf("Running %d Runs starting at %d\n", runs, run)
	rc := &ss.Loops.GetLoop(etime.Train, etime.Run).Counter
	rc.Set(run)
	rc.Max = run + runs

	ss.Loops.GetLoop(etime.Train, etime.Epoch).Counter.Max = ss.Args.Int("epochs")

	ss.NewRun()
	ss.Loops.Run()

	ss.Logs.CloseLogFiles()

	if netdata {
		ss.GUI.SaveNetData(ss.Stats.String("RunName"))
	}

	ss.MPIFinalize()
}

////////////////////////////////////////////////////////////////////
//  MPI code

// MPIInit initializes MPI
func (ss *Sim) MPIInit() {
	mpi.Init()
	var err error
	ss.Comm, err = mpi.NewComm(nil) // use all procs
	if err != nil {
		log.Println(err)
		ss.Args.SetBool("mpi", false)
	} else {
		mpi.Printf("MPI running on %d procs\n", mpi.WorldSize())
	}
}

// MPIFinalize finalizes MPI
func (ss *Sim) MPIFinalize() {
	if ss.Args.Bool("mpi") {
		mpi.Finalize()
	}
}

// CollectDWts collects the weight changes from all synapses into AllDWts
// includes all other long adapting factors too: DTrgAvg, ActAvg, etc
func (ss *Sim) CollectDWts(net *axon.Network) {
	net.CollectDWts(&ss.AllDWts)
}

// MPIWtFmDWt updates weights from weight changes, using MPI to integrate
// DWt changes across parallel nodes, each of which are learning on different
// sequences of inputs.
func (ss *Sim) MPIWtFmDWt() {
	if ss.Args.Bool("mpi") {
		ss.CollectDWts(ss.Net)
		ndw := len(ss.AllDWts)
		if len(ss.SumDWts) != ndw {
			ss.SumDWts = make([]float32, ndw)
		}
		ss.Comm.AllReduceF32(mpi.OpSum, ss.SumDWts, ss.AllDWts)
		ss.Net.SetDWts(ss.SumDWts, mpi.WorldSize())
	}
	ss.Net.WtFmDWt(&ss.Time)
}

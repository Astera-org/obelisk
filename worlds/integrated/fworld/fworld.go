// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package main

// FWorld is a simple Flat World where an agent can look around for food or water.
// See the main logic with ctrl-f "Connect to Server".

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	log "github.com/Astera-org/easylog"
	"github.com/Astera-org/models/agent"
	"github.com/Astera-org/models/library/metrics"
	"github.com/Astera-org/worlds/network"
	net_env "github.com/Astera-org/worlds/network/gengo/env"
	"github.com/Astera-org/worlds/network_agent"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/evec"
	"github.com/emer/emergent/popcode"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/etview"
	"github.com/emer/etable/metric"
	"github.com/goki/gi/colormap"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/gist"
	"github.com/goki/gi/giv"
	"github.com/goki/ki/ints"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"strconv"
)

// FWorld is a flat-world grid-based environment
type FWorld struct {
	agent.WorldInterface
	Nm           string                      `desc:"name of this environment"`
	Dsc          string                      `desc:"description of this environment"`
	Disp         bool                        `desc:"update display -- turn off to make it faster"`
	Size         evec.Vec2i                  `desc:"size of 2D world"`
	PatSize      evec.Vec2i                  `desc:"size of patterns for mats, acts"`
	World        *etensor.Int                `view:"no-inline" desc:"2D grid world, each cell is a material (mat)"`
	PctCortex    float64                     `desc:"proportion of action driven by the cortex vs. hard-coded reflexive subcortical"`
	PctCortexMax float64                     `desc:"maximum PctCortex, when running on the schedule"`
	Mats         []string                    `desc:"list of materials in the world, 0 = empty.  Any superpositions of states (e.g., CoveredFood) need to be discretely encoded, can be transformed through action rules"`
	MatColors    []string                    `desc:"color strings in material order"`
	MatMap       map[string]int              `desc:"map of material name to index stored in world cell"`
	Trace        *etensor.Int                `view:"no-inline" desc:"trace of movement for visualization"`
	WorldWin     *gi.Window                  `view:"-" desc:"FWorld GUI window"`
	WorldTabs    *gi.TabView                 `view:"-" desc:"FWorld TabView"`
	TraceView    *etview.TensorGrid          `desc:"view of the activity trace"`
	WorldView    *etview.TensorGrid          `desc:"view of the world"`
	IsRunning    bool                        `view:"-" desc:"true if sim is running"`
	BarrierIdx   int                         `desc:"index of material below which (inclusive) cannot move -- e.g., 1 for wall"`
	Pats         map[string]*etensor.Float32 `desc:"patterns for each material (must include Empty) and for each action"`
	ActPats      map[string]*etensor.Float32 `desc:"patterns for each action -- for decoding"`
	Acts         []string                    `desc:"list of actions: starts with: Stay, Left, Right, Forward, Back, then extensible"`
	ActMap       map[string]int              `desc:"action map of action names to indexes"`
	Inters       []string                    `desc:"list of interoceptive body states, represented as pop codes"`
	InterMap     map[string]int              `desc:"map of interoceptive state names to indexes"`
	Params       map[string]float32          `desc:"map of optional interoceptive and world-dynamic parameters -- cleaner to store in a map"`
	FOV          int                         `desc:"field of view in degrees, e.g., 180, must be even multiple of AngInc"`
	AngInc       int                         `desc:"angle increment for rotation, in degrees -- defaults to 15"`
	NRotAngles   int                         `inactive:"+" desc:"total number of rotation angles in a circle"`
	NFOVRays     int                         `inactive:"+" desc:"total number of FOV rays that are traced"`
	WallUrgency  float32                     `desc:"urgency when right against a wall"`
	EatUrgency   float32                     `desc:"urgency for eating and drinking"`
	CloseUrgency float32                     `desc:"urgency for being close to food / water"`
	FwdMargin    float32                     `desc:"forward action must be this factor larger than 2nd best option to be selected"`
	ShowRays     bool                        `desc:"for debugging only: show the main depth rays as they are traced out from point"`
	ShowFovRays  bool                        `desc:"for debugging only: show the fovea rays as they are traced out from point"`
	TraceActGen  bool                        `desc:"for debugging, print out a trace of the action generation logic"`
	FoveaSize    int                         `desc:"number of items on each size of the fovea, in addition to center (0 or more)"`
	FoveaAngInc  int                         `desc:"scan angle for fovea"`
	PopSize      int                         `inactive:"+" desc:"number of units in population codes"`
	PopCode      popcode.OneD                `desc:"generic population code values, in normalized units"`
	DepthSize    int                         `inactive:"+" desc:"number of units in depth population codes"`
	DepthPools   int                         `inactive:"+" desc:"number of pools to divide DepthSize into"`
	DepthCode    popcode.OneD                `desc:"population code for depth, in normalized units"`
	AngCode      popcode.Ring                `desc:"angle population code values, in normalized units"`

	// current state below (params above)
	PosF          mat32.Vec2                  `inactive:"+" desc:"current location of agent, floating point"`
	PosI          evec.Vec2i                  `inactive:"+" desc:"current location of agent, integer"`
	Angle         int                         `inactive:"+" desc:"current angle, in degrees"`
	RotAng        int                         `inactive:"+" desc:"angle that we just rotated -- drives vestibular"`
	Urgency       float32                     `inactive:"+" desc:"for ActGen, level of urgency for following the generated action"`
	Act           int                         `inactive:"+" desc:"last action taken"`
	Depths        []float32                   `desc:"depth for each angle (NFOVRays), raw"`
	DepthLogs     []float32                   `desc:"depth for each angle (NFOVRays), normalized log"`
	ViewMats      []int                       `inactive:"+" desc:"material at each angle"`
	FovMats       []int                       `desc:"materials at fovea, L-R"`
	FovDepths     []float32                   `desc:"raw depths to foveal materials, L-R"`
	FovDepthLogs  []float32                   `desc:"normalized log depths to foveal materials, L-R"`
	ProxMats      []int                       `desc:"material at each right angle: front, left, right back"`
	ProxPos       []evec.Vec2i                `desc:"coordinates for proximal grid points: front, left, right, back"`
	InterStates   map[string]float32          `inactive:"+" desc:"floating point value of internal states -- dim of Inters"`
	CurStates     map[string]*etensor.Float32 `desc:"current rendered state tensors -- extensible map"`
	NextStates    map[string]*etensor.Float32 `desc:"next rendered state tensors -- updated from actions"`
	RefreshEvents map[int]*WEvent             `desc:"list of events, key is tick step, to check each step to drive refresh of consumables -- removed from this active list when complete"`
	AllEvents     map[int]*WEvent             `desc:"list of all events, key is tick step"`
	Run           env.Ctr                     `view:"inline" desc:"current run of model as provided during Init"`
	Epoch         env.Ctr                     `view:"inline" desc:"increments over arbitrary fixed number of trials, for general stats-tracking"`
	Trial         env.Ctr                     `view:"inline" desc:"increments for each step of world, loops over epochs -- for general stats-tracking independent of env state"`
	Tick          env.Ctr                     `view:"monolithic time counter -- counts up time every step -- used for refreshing world state"`
	Event         env.Ctr                     `view:"arbitrary counter for steps within a scene -- resets at consumption event"`
	Scene         env.Ctr                     `view:"arbitrary counter incrementing over a coherent sequence of events: e.g., approaching food -- increments at consumption"`
	Episode       env.Ctr                     `view:"arbitrary counter incrementing over scenes within larger episode: feeding, drinking, exploring, etc"`

	// State variables that help with control flow.
	paused  bool
	stepOne bool

	LastActionPredicted int     `view:"-" desc:"last action suggested by model over network"`
	UseGUI              bool    `view:"determines if gui is going to be used or not"`
	predictedActions    []int32 `view:"-" desc:"for scoring counts what action was taken on a particular trial"`
	bestActions         []int32 `view:"-" desc:"for scoring, counts what action was taken on a particular trial"`
}

var KiT_FWorld = kit.Types.AddType(&FWorld{}, FWorldProps)

func (ev *FWorld) Name() string { return ev.Nm }
func (ev *FWorld) Desc() string { return ev.Dsc }

// Config configures the world
func (ev *FWorld) Config(ntrls int) {
	ev.predictedActions = make([]int32, 1)
	ev.bestActions = make([]int32, 1)
	ev.Nm = "Demo"
	ev.Dsc = "Example world with basic food / water / eat / drink actions"
	ev.Disp = true
	ev.Mats = []string{"Empty", "Wall", "Food", "Water", "FoodWas", "WaterWas"}
	ev.BarrierIdx = 1
	ev.Acts = []string{"Forward", "Left", "Right", "Eat", "Drink"} // "Stay", "Backward",
	ev.Inters = []string{"Energy", "Hydra", "BumpPain", "FoodRew", "WaterRew"}

	ev.Params = make(map[string]float32)

	ev.Params["TimeCost"] = 0.001  // decrement due to existing for 1 unit of time, in energy and hydration
	ev.Params["MoveCost"] = 0.002  // additional decrement due to moving
	ev.Params["RotCost"] = 0.001   // additional decrement due to rotating one step
	ev.Params["BumpCost"] = 0.01   // additional decrement in addition to move cost, for bumping into things
	ev.Params["EatCost"] = 0.005   // additional decrement in hydration due to eating
	ev.Params["DrinkCost"] = 0.005 // additional decrement in energy due to drinking
	ev.Params["EatVal"] = 0.9      // increment in energy due to eating one unit of food
	ev.Params["DrinkVal"] = 0.9    // increment in hydration due to drinking one unit of water
	ev.Params["FoodRefresh"] = 100 // time steps before food is refreshed
	ev.Params["WaterRefresh"] = 50 // time steps before water is refreshed

	ev.Disp = true
	ev.Size.Set(100, 100)
	ev.PatSize.Set(5, 5)
	ev.AngInc = 15
	ev.FOV = 180
	ev.FoveaSize = 1
	ev.FoveaAngInc = 5
	ev.WallUrgency = .9
	ev.EatUrgency = .8
	ev.CloseUrgency = .5
	ev.FwdMargin = 2
	ev.PopSize = 16
	ev.PopCode.Defaults()
	ev.PopCode.SetRange(-0.2, 1.2, 0.1)
	ev.DepthSize = 32
	ev.DepthPools = 8
	ev.DepthCode.Defaults()
	ev.DepthCode.SetRange(0.1, 1, 0.05)
	ev.AngCode.Defaults()
	ev.AngCode.SetRange(0, 1, 0.1)

	// debugging options:
	ev.ShowRays = false
	ev.ShowFovRays = false
	ev.TraceActGen = false

	ev.Trial.Max = ntrls

	ev.ConfigPats()
	ev.ConfigImpl()

	// uncomment to generate a new world
	ev.GenWorld()
	ev.SaveWorld("world.tsv")
}

// ConfigPats configures the bit pattern representations of mats and acts
func (ev *FWorld) ConfigPats() {
	ev.Pats = make(map[string]*etensor.Float32)
	ev.ActPats = make(map[string]*etensor.Float32)
	for _, m := range ev.Mats {
		t := &etensor.Float32{}
		t.SetShape([]int{ev.PatSize.Y, ev.PatSize.X}, nil, []string{"Y", "X"})
		ev.Pats[m] = t
	}
	for _, a := range ev.Acts {
		t := &etensor.Float32{}
		t.SetShape([]int{ev.PatSize.Y, ev.PatSize.X}, nil, []string{"Y", "X"})
		ev.Pats[a] = t
	}

	ev.OpenPats("pats.json") // TODO add to the .cfg

	for _, a := range ev.Acts {
		ev.ActPats[a] = ev.Pats[a]
	}
}

// ConfigImpl does the automatic parts of configuration
// generally does not require editing
func (ev *FWorld) ConfigImpl() {
	ev.NFOVRays = (ev.FOV / ev.AngInc) + 1
	ev.NRotAngles = (360 / ev.AngInc) + 1

	ev.World = &etensor.Int{}
	ev.World.SetShape([]int{ev.Size.Y, ev.Size.X}, nil, []string{"Y", "X"})

	ev.ProxMats = make([]int, 4)
	ev.ProxPos = make([]evec.Vec2i, 4)

	ev.CurStates = make(map[string]*etensor.Float32)
	ev.NextStates = make(map[string]*etensor.Float32)

	dv := &etensor.Float32{}
	dv.SetShape([]int{ev.DepthPools, ev.NFOVRays, ev.DepthSize / ev.DepthPools, 1}, nil, []string{"Pools", "Angle", "Pop", "1"})
	ev.NextStates["Depth"] = dv

	dv = &etensor.Float32{}
	dv.SetShape([]int{1, ev.NFOVRays, ev.DepthSize, 1}, nil, []string{"1", "Angle", "Pop", "1"})
	ev.NextStates["DepthRender"] = dv

	ev.Depths = make([]float32, ev.NFOVRays)
	ev.DepthLogs = make([]float32, ev.NFOVRays)
	ev.ViewMats = make([]int, ev.NFOVRays)

	fsz := 1 + 2*ev.FoveaSize
	fd := &etensor.Float32{}
	fd.SetShape([]int{ev.DepthPools, fsz, ev.DepthSize / ev.DepthPools, 1}, nil, []string{"Pools", "Angle", "Pop", "1"})
	ev.NextStates["FovDepth"] = fd

	fd = &etensor.Float32{}
	fd.SetShape([]int{1, fsz, ev.DepthSize, 1}, nil, []string{"1", "Angle", "Pop", "1"})
	ev.NextStates["FovDepthRender"] = fd

	fv := &etensor.Float32{}
	fv.SetShape([]int{1, fsz, ev.PatSize.Y, ev.PatSize.X}, nil, []string{"1", "Angle", "Y", "X"})
	ev.NextStates["Fovea"] = fv

	ps := &etensor.Float32{}
	ps.SetShape([]int{1, 4, 2, 1}, nil, []string{"1", "Pos", "OnOff", "1"})
	ev.NextStates["ProxSoma"] = ps

	vs := &etensor.Float32{}
	vs.SetShape([]int{1, 2, ev.PopSize, 1}, nil, []string{"1", "RotAng", "Pop", "1"})
	ev.NextStates["Vestibular"] = vs

	is := &etensor.Float32{}
	is.SetShape([]int{1, len(ev.Inters), ev.PopSize, 1}, nil, []string{"1", "Inters", "Pop", "1"})
	ev.NextStates["Inters"] = is

	av := &etensor.Float32{}
	av.SetShape([]int{ev.PatSize.Y, ev.PatSize.X}, nil, []string{"Y", "X"})
	ev.NextStates["Action"] = av

	ev.CopyNextToCur() // get CurStates from NextStates

	ev.FovMats = make([]int, fsz)
	ev.FovDepths = make([]float32, fsz)
	ev.FovDepthLogs = make([]float32, fsz)

	ev.MatMap = make(map[string]int, len(ev.Mats))
	for i, m := range ev.Mats {
		ev.MatMap[m] = i
	}
	ev.ActMap = make(map[string]int, len(ev.Acts))
	for i, m := range ev.Acts {
		ev.ActMap[m] = i
	}
	ev.InterMap = make(map[string]int, len(ev.Inters))
	for i, m := range ev.Inters {
		ev.InterMap[m] = i
	}
	ev.InterStates = make(map[string]float32, len(ev.Inters))
	for _, m := range ev.Inters {
		ev.InterStates[m] = 0
	}

	ev.Run.Scale = env.Run
	ev.Epoch.Scale = env.Epoch
	ev.Trial.Scale = env.Trial
	ev.Tick.Scale = env.Tick
	ev.Event.Scale = env.Event
	ev.Scene.Scale = env.Scene
	ev.Episode.Scale = env.Episode
}

func (ev *FWorld) Validate() error {
	if ev.Size.IsNil() {
		return fmt.Errorf("FWorld: %v has size == 0 -- need to Config", ev.Nm)
	}
	return nil
}

func (ev *FWorld) State(element string) etensor.Tensor {
	return ev.CurStates[element]
}

// String returns the current state as a string
func (ev *FWorld) String() string {
	return fmt.Sprintf("Evt_%d_Pos_%d_%d_Ang_%d_Act_%s", ev.Event.Cur, ev.PosI.X, ev.PosI.Y, ev.Angle, ev.Acts[ev.Act])
}

func (ev *FWorld) InitWorld(details map[string]string) {
	ev.Init(0)
}

// Init is called to restart environment
func (ev *FWorld) Init(run int) {

	// note: could gen a new random world too..
	ev.OpenWorld("world.tsv")

	ev.Run.Init()
	ev.Epoch.Init()
	ev.Trial.Init()
	ev.Tick.Init()
	ev.Event.Init()
	ev.Scene.Init()
	ev.Episode.Init()

	ev.Run.Cur = run
	ev.Trial.Cur = -1 // init state -- key so that first Step() = 0
	ev.Tick.Cur = -1
	ev.Event.Cur = -1

	ev.PosI = ev.Size.DivScalar(2) // start in middle -- could be random..
	ev.PosF = ev.PosI.ToVec2()
	for i := 0; i < 4; i++ {
		ev.ProxMats[i] = 0
	}

	ev.Angle = 0
	ev.RotAng = 0
	ev.InterStates["Energy"] = 1
	ev.InterStates["Hydra"] = 1
	ev.InterStates["BumpPain"] = 0
	ev.InterStates["FoodRew"] = 0
	ev.InterStates["WaterRew"] = 0

	ev.RefreshEvents = make(map[int]*WEvent)
	ev.AllEvents = make(map[int]*WEvent)
}

// SetWorld sets given mat at given point coord in world
func (ev *FWorld) SetWorld(p evec.Vec2i, mat int) {
	ev.World.Set([]int{p.Y, p.X}, mat)
}

// GetWorld returns mat at given point coord in world
func (ev *FWorld) GetWorld(p evec.Vec2i) int {
	return ev.World.Value([]int{p.Y, p.X})
}

////////////////////////////////////////////////////////////////////
// I/O

// SaveWorld saves the world to a tsv file with empty string for empty cells
func (ev *FWorld) SaveWorld(filename gi.FileName) error {
	fp, err := os.Create(string(filename))
	if err != nil {
		log.Error("Error creating file:", err)
		return err
	}
	defer fp.Close()
	bw := bufio.NewWriter(fp)
	for y := 0; y < ev.Size.Y; y++ {
		for x := 0; x < ev.Size.X; x++ {
			mat := ev.World.Value([]int{y, x})
			ms := ev.Mats[mat]
			if ms == "Empty" {
				ms = ""
			}
			bw.WriteString(ms + "\t")
		}
		bw.WriteString("\n")
	}
	bw.Flush()
	return nil
}

// OpenWorld loads the world from a tsv file with empty string for empty cells
func (ev *FWorld) OpenWorld(filename gi.FileName) error {
	fp, err := os.Open(string(filename))
	if err != nil {
		log.Error("Error opening file:", err)
		return err
	}
	defer fp.Close()
	ev.World.SetZeros()
	scan := bufio.NewScanner(fp)
	for y := 0; y < ev.Size.Y; y++ {
		if !scan.Scan() {
			break
		}
		ln := scan.Bytes()
		sz := len(ln)
		if sz == 0 {
			break
		}
		sp := bytes.Split(ln, []byte("\t"))
		sz = ints.MinInt(ev.Size.X, len(sp)-1)
		for x := 0; x < ev.Size.X; x++ {
			ms := string(sp[x])
			if ms == "" {
				continue
			}
			mi, ok := ev.MatMap[ms]
			if !ok {
				log.Info("Mat not found: %s\n", ms)
			} else {
				ev.World.Set([]int{y, x}, mi)
			}
		}
	}
	return nil
}

// SavePats saves the patterns
func (ev *FWorld) SavePats(filename gi.FileName) error {
	jenc, _ := json.MarshalIndent(ev.Pats, "", " ")
	return ioutil.WriteFile(string(filename), jenc, 0644)
}

// OpenPats opens the patterns
func (ev *FWorld) OpenPats(filename gi.FileName) error {
	fp, err := os.Open(string(filename))
	if err != nil {
		log.Error("ERROR! Can't find pats file! This will prevent the model from working! :", err)
		return err
	}
	defer fp.Close()
	b, err := ioutil.ReadAll(fp)
	err = json.Unmarshal(b, &ev.Pats)
	if err != nil {
		log.Error("", err)
	}
	return err
}

// AngMod returns angle modulo within 360 degrees
func AngMod(ang int) int {
	if ang < 0 {
		ang += 360
	} else if ang > 360 {
		ang -= 360
	}
	return ang
}

// AngVec returns the incremental vector to use for given angle, in deg
// such that the largest value is 1.
func AngVec(ang int) mat32.Vec2 {
	a := mat32.DegToRad(float32(AngMod(ang)))
	v := mat32.Vec2{mat32.Cos(a), mat32.Sin(a)}
	return NormVecLine(v)
}

// NormVec normalize vector for drawing a line
func NormVecLine(v mat32.Vec2) mat32.Vec2 {
	av := v.Abs()
	if av.X > av.Y {
		v = v.DivScalar(av.X)
	} else {
		v = v.DivScalar(av.Y)
	}
	return v
}

// NextVecPoint returns the next grid point along vector,
// from given current floating and grid points.  v is normalized
// such that the largest value is 1.
func NextVecPoint(cp, v mat32.Vec2) (mat32.Vec2, evec.Vec2i) {
	n := cp.Add(v)
	g := evec.NewVec2iFmVec2Round(n)
	return n, g
}

////////////////////////////////////////////////////////////////////
// Vision

// ScanDepth does simple ray-tracing to find depth and material along each angle vector
func (ev *FWorld) ScanDepth() {
	nmat := len(ev.Mats)
	_ = nmat
	idx := 0
	hang := ev.FOV / 2
	maxld := mat32.Log(1 + mat32.Sqrt(float32(ev.Size.X*ev.Size.X+ev.Size.Y*ev.Size.Y)))
	for ang := hang; ang >= -hang; ang -= ev.AngInc {
		v := AngVec(ang + ev.Angle)
		op := ev.PosF
		cp := op
		gp := evec.Vec2i{}
		depth := float32(-1)
		vmat := 0 // first non-empty visible material
		for {
			cp, gp = NextVecPoint(cp, v)
			if gp.X < 0 || gp.X >= ev.Size.X {
				break
			}
			if gp.Y < 0 || gp.Y >= ev.Size.Y {
				break
			}
			mat := ev.GetWorld(gp)
			if mat > 0 && mat <= ev.BarrierIdx {
				vmat = mat
				depth = cp.DistTo(op)
				break
			}
			if ev.ShowRays {
				ev.SetWorld(gp, nmat+idx*2) // visualization
			}
		}
		ev.Depths[idx] = depth
		ev.ViewMats[idx] = vmat
		if depth > 0 {
			ev.DepthLogs[idx] = mat32.Log(1+depth) / maxld
		} else {
			ev.DepthLogs[idx] = 1
		}
		idx++
	}
}

// ScanFovea does simple ray-tracing to find depth and material for fovea
func (ev *FWorld) ScanFovea() {
	nmat := len(ev.Mats)
	idx := 0
	maxld := mat32.Log(1 + mat32.Sqrt(float32(ev.Size.X*ev.Size.X+ev.Size.Y*ev.Size.Y)))
	for fi := -ev.FoveaSize; fi <= ev.FoveaSize; fi++ {
		ang := -fi * ev.FoveaAngInc
		v := AngVec(ang + ev.Angle)
		op := ev.PosF
		cp := op
		gp := evec.Vec2i{}
		depth := float32(-1)
		vmat := 0 // first non-empty visible material
		for {
			cp, gp = NextVecPoint(cp, v)
			if gp.X < 0 || gp.X >= ev.Size.X {
				break
			}
			if gp.Y < 0 || gp.Y >= ev.Size.Y {
				break
			}
			mat := ev.GetWorld(gp)
			if mat > 0 && mat < nmat {
				vmat = mat
				depth = cp.DistTo(op)
				break
			}
			if ev.ShowFovRays {
				ev.SetWorld(gp, nmat+idx*2) // visualization
			}
		}
		ev.FovDepths[idx] = depth
		ev.FovMats[idx] = vmat
		if depth > 0 {
			ev.FovDepthLogs[idx] = mat32.Log(1+depth) / maxld
		} else {
			ev.FovDepthLogs[idx] = 1
		}
		idx++
	}
}

// ScanProx scan the proximal space around the agent
func (ev *FWorld) ScanProx() {
	angs := []int{0, -90, 90, 180}
	for i := 0; i < 4; i++ {
		v := AngVec(ev.Angle + angs[i])
		_, gp := NextVecPoint(ev.PosF, v)
		ev.ProxMats[i] = ev.GetWorld(gp)
		ev.ProxPos[i] = gp
	}
}

// IncState increments state by factor, keeping bounded between 0-1
func (ev *FWorld) IncState(nm string, inc float32) {
	st := ev.InterStates[nm]
	st += inc
	st = mat32.Max(st, 0)
	st = mat32.Min(st, 1)
	ev.InterStates[nm] = st
}

// PassTime does effects of time, initializes rewards
func (ev *FWorld) PassTime() {
	ev.Scene.Same()
	tc := ev.Params["TimeCost"]
	ev.IncState("Energy", -tc)
	ev.IncState("Hydra", -tc)
	ev.InterStates["BumpPain"] = 0
	ev.InterStates["FoodRew"] = 0
	ev.InterStates["WaterRew"] = 0
}

////////////////////////////////////////////////////////////////////
// Actions

// WEvent records an event
type WEvent struct {
	Tick   int        `desc:"tick when event happened"`
	PosI   evec.Vec2i `desc:"discrete integer grid position where event happened"`
	PosF   mat32.Vec2 `desc:"floating point grid position where event happened"`
	Angle  int        `desc:"angle pointing when event happened"`
	Act    int        `desc:"action that took place"`
	Mat    int        `desc:"material that was involved (front fovea mat)"`
	MatPos evec.Vec2i `desc:"position of material involved in event"`
}

// NewEvent returns new event with current state and given act, mat
func (ev *FWorld) NewEvent(act, mat int, matpos evec.Vec2i) *WEvent {
	return &WEvent{Tick: ev.Tick.Cur, PosI: ev.PosI, PosF: ev.PosF, Angle: ev.Angle, Act: act, Mat: mat, MatPos: matpos}
}

// AddNewEventRefresh adds event to RefreshEvents (a consumable was consumed).
// always adds to AllEvents
func (ev *FWorld) AddNewEventRefresh(wev *WEvent) {
	ev.RefreshEvents[wev.Tick] = wev
	ev.AllEvents[wev.Tick] = wev
}

// RefreshWorld refreshes consumables
func (ev *FWorld) RefreshWorld() {
	ct := ev.Tick.Cur
	fr := int(ev.Params["FoodRefresh"])
	wr := int(ev.Params["WaterRefresh"])
	fmat := ev.MatMap["Food"]
	wmat := ev.MatMap["Water"]
	for t, wev := range ev.RefreshEvents {
		setmat := 0
		switch wev.Mat {
		case fmat:
			if t+fr < ct {
				setmat = fmat
			}
		case wmat:
			if t+wr < ct {
				setmat = wmat
			}
		}
		if setmat != 0 {
			ev.SetWorld(wev.MatPos, setmat)
			delete(ev.RefreshEvents, t)
		}
	}
}

// DecodeAct decodes action from given tensor of activation states
// Forward is only selected if it is 2x larger than other options
func (ev *FWorld) DecodeAct(vt etensor.Float32) int {
	cnm := ""
	var dst, fdst float32
	for nm, pat := range ev.ActPats {
		d := metric.Correlation32(vt.Values, pat.Values)
		if nm == "Forward" {
			fdst = d
		} else {
			if cnm == "" || d > dst {
				cnm = nm
				dst = d
			}
		}
	}
	if fdst > ev.FwdMargin*dst { // only if x bigger
		cnm = "Forward"
	}
	act, ok := ev.ActMap[cnm]
	if !ok {
		act = rand.Intn(len(ev.Acts))
	}
	return act
}

// TakeAct takes the action, updates state
func (ev *FWorld) TakeAct(act int) {
	as := ""
	if act >= len(ev.Acts) || act < 0 {
		as = "Stay"
	} else {
		as = ev.Acts[act]
	}
	ev.PassTime()

	ev.RotAng = 0

	nmat := len(ev.Mats)
	frmat := ints.MinInt(ev.ProxMats[0], nmat)
	behmat := ev.ProxMats[3] // behind
	front := ev.Mats[frmat]  // state in front

	mvc := ev.Params["MoveCost"]
	rotc := ev.Params["RotCost"]
	bumpc := ev.Params["BumpCost"]

	ecost := float32(0) // extra energy cost
	hcost := float32(0) // extra hydra cost

	switch as {
	case "Stay":
	case "Left":
		ev.RotAng = ev.AngInc
		ev.Angle = AngMod(ev.Angle + ev.RotAng)
		ecost = rotc
		hcost = rotc
	case "Right":
		ev.RotAng = -ev.AngInc
		ev.Angle = AngMod(ev.Angle + ev.RotAng)
		ecost = rotc
		hcost = rotc
	case "Forward":
		ecost = mvc
		hcost = mvc
		if frmat > 0 && frmat <= ev.BarrierIdx {
			ev.InterStates["BumpPain"] = 1
			ecost += bumpc
			hcost += bumpc
		} else {
			ev.PosF, ev.PosI = NextVecPoint(ev.PosF, AngVec(ev.Angle))
		}
	case "Backward":
		ecost = mvc
		hcost = mvc
		if behmat > 0 && behmat <= ev.BarrierIdx {
			ev.InterStates["BumpPain"] = 1
			ecost += bumpc
			hcost += bumpc
		} else {
			ev.PosF, ev.PosI = NextVecPoint(ev.PosF, AngVec(AngMod(ev.Angle+180)))
		}
	case "Eat":
		if front == "Food" {
			ev.InterStates["FoodRew"] = 1
			hcost += ev.Params["EatCost"]
			ecost -= ev.Params["EatVal"]
			ev.AddNewEventRefresh(ev.NewEvent(act, frmat, ev.ProxPos[0]))
			ev.SetWorld(ev.ProxPos[0], ev.MatMap["FoodWas"])
			ev.Event.Set(0)
			ev.Scene.Incr()
		}
	case "Drink":
		if front == "Water" {
			ev.InterStates["WaterRew"] = 1
			ecost += ev.Params["DrinkCost"]
			hcost -= ev.Params["DrinkVal"]
			ev.AddNewEventRefresh(ev.NewEvent(act, frmat, ev.ProxPos[0]))
			ev.SetWorld(ev.ProxPos[0], ev.MatMap["WaterWas"])
			ev.Event.Set(0)
			ev.Scene.Incr()
		}
	}
	ev.ScanDepth()
	ev.ScanFovea()
	ev.ScanProx()

	ev.IncState("Energy", -ecost)
	ev.IncState("Hydra", -hcost)

	ev.RenderState()
}

// RenderView renders the current view state to NextStates tensor input states
func (ev *FWorld) RenderView() {
	dv := ev.NextStates["Depth"]
	dvr := ev.NextStates["DepthRender"]
	np := ev.DepthSize / ev.DepthPools
	for i := 0; i < ev.NFOVRays; i++ {
		sv := dvr.SubSpace([]int{0, i}).(*etensor.Float32)
		ev.DepthCode.Encode(&sv.Values, ev.DepthLogs[i], ev.DepthSize, popcode.Set)
		for dp := 0; dp < ev.DepthPools; dp++ {
			for pi := 0; pi < np; pi++ {
				ri := dp*np + pi
				dv.Set([]int{dp, i, pi, 0}, sv.Values[ri])
			}
		}
	}

	fsz := 1 + 2*ev.FoveaSize
	fd := ev.NextStates["FovDepth"]
	fdr := ev.NextStates["FovDepthRender"]
	fv := ev.NextStates["Fovea"]
	for i := 0; i < fsz; i++ {
		sv := fdr.SubSpace([]int{0, i}).(*etensor.Float32)
		ev.DepthCode.Encode(&sv.Values, ev.FovDepthLogs[i], ev.DepthSize, popcode.Set)
		for dp := 0; dp < ev.DepthPools; dp++ {
			for pi := 0; pi < np; pi++ {
				ri := dp*np + pi
				fd.Set([]int{dp, i, pi, 0}, sv.Values[ri])
			}
		}

		fm := ev.FovMats[i]
		if fm < len(ev.Mats) {
			sv := fv.SubSpace([]int{0, i}).(*etensor.Float32)
			ms := ev.Mats[fm]
			mp, ok := ev.Pats[ms]
			if ok {
				sv.CopyFrom(mp)
			}
		}
	}
}

// RenderProxSoma renders proximal soma state
func (ev *FWorld) RenderProxSoma() {
	ps := ev.NextStates["ProxSoma"]
	ps.SetZeros()
	for i := 0; i < 4; i++ {
		if ev.ProxMats[i] != 0 {
			ps.Set([]int{0, i, 0, 0}, 1) // on
		} else {
			ps.Set([]int{0, i, 1, 0}, 1) // off
		}
	}
}

// RenderInters renders interoceptive state
func (ev *FWorld) RenderInters() {
	is := ev.NextStates["Inters"]
	for k, v := range ev.InterStates {
		idx := ev.InterMap[k]
		sv := is.SubSpace([]int{0, idx}).(*etensor.Float32)
		ev.PopCode.Encode(&sv.Values, v, ev.PopSize, popcode.Set)
	}
}

// RenderVestibular renders vestibular state
func (ev *FWorld) RenderVestibular() {
	vs := ev.NextStates["Vestibular"]
	sv := vs.SubSpace([]int{0, 0}).(*etensor.Float32)
	nv := 0.5*(float32(-ev.RotAng)/15.0) + 0.5
	ev.PopCode.Encode(&sv.Values, nv, ev.PopSize, popcode.Set)

	sv = vs.SubSpace([]int{0, 1}).(*etensor.Float32)
	nv = (float32(ev.Angle) / 360.0)
	ev.AngCode.Encode(&sv.Values, nv, ev.PopSize)
}

// RenderAction renders action pattern
func (ev *FWorld) RenderAction() {
	av := ev.NextStates["Action"]
	if ev.Act < len(ev.Acts) {
		as := ev.Acts[ev.Act]
		ap, ok := ev.Pats[as]
		if ok {
			av.CopyFrom(ap)
		}
	}
}

// RenderState renders the current state into NextState vars //primary point for converting the state
func (ev *FWorld) RenderState() {
	ev.RenderView()
	ev.RenderProxSoma()
	ev.RenderInters()
	ev.RenderVestibular()
	ev.RenderAction()
}

// CopyNextToCur copy next state to current state
func (ev *FWorld) CopyNextToCur() {
	for k, ns := range ev.NextStates {
		cs, ok := ev.CurStates[k]
		if !ok {
			cs = ns.Clone().(*etensor.Float32)
			ev.CurStates[k] = cs
		} else {
			cs.CopyFrom(ns)
		}
	}
}

// Step is called to advance the environment state
func (ev *FWorld) Step() bool {
	ev.Epoch.Same() // good idea to just reset all non-inner-most counters at start
	ev.CopyNextToCur()
	ev.Tick.Incr()
	ev.Event.Incr()
	ev.RefreshWorld()
	if ev.Trial.Incr() { // true if wraps around Max back to 0
		ev.Epoch.Incr()
	}
	return true
}

func (ev *FWorld) Action(action string, nop etensor.Tensor) {
	a, ok := ev.ActMap[action]
	if !ok {
		fmt.Printf("Action not recognized: %s\n", action)
		return
	}
	ev.Act = a
	ev.TakeAct(ev.Act)
}

func (ev *FWorld) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
	switch scale {
	case env.Run:
		return ev.Run.Query()
	case env.Epoch:
		return ev.Epoch.Query()
	case env.Trial:
		return ev.Trial.Query()
	case env.Tick:
		return ev.Tick.Query()
	case env.Event:
		return ev.Event.Query()
	case env.Scene:
		return ev.Scene.Query()
	case env.Episode:
		return ev.Episode.Query()
	}
	return -1, -1, false
}

// Compile-time check that implements Env interface
var _ env.Env = (*FWorld)(nil)

var FWorldProps = ki.Props{
	"ToolBar": ki.PropSlice{
		{"OpenWorld", ki.Props{
			"label": "Open World...",
			"icon":  "file-open",
			"desc":  "Open World from tsv file",
			"Args": ki.PropSlice{
				{"File Name", ki.Props{
					"ext": ".tsv",
				}},
			},
		}},
		{"SaveWorld", ki.Props{
			"label": "Save World...",
			"icon":  "file-save",
			"desc":  "Save World to tsv file",
			"Args": ki.PropSlice{
				{"File Name", ki.Props{
					"ext": ".tsv",
				}},
			},
		}},
		{"OpenPats", ki.Props{
			"label": "Open Pats...",
			"icon":  "file-open",
			"desc":  "Open pats from json file",
			"Args": ki.PropSlice{
				{"File Name", ki.Props{
					"ext": ".json",
				}},
			},
		}},
		{"SavePats", ki.Props{
			"label": "Save Pats...",
			"icon":  "file-save",
			"desc":  "Save pats to json file",
			"Args": ki.PropSlice{
				{"File Name", ki.Props{
					"ext": ".json",
				}},
			},
		}},
	},
}

////////////////////////////////////////////////////////////////////
// Render world

// WorldLineHoriz draw horizontal line
func (ev *FWorld) WorldLineHoriz(st, ed evec.Vec2i, mat int) {
	sx := ints.MinInt(st.X, ed.X)
	ex := ints.MaxInt(st.X, ed.X)
	for x := sx; x <= ex; x++ {
		ev.World.Set([]int{st.Y, x}, mat)
	}
}

// WorldLineVert draw vertical line
func (ev *FWorld) WorldLineVert(st, ed evec.Vec2i, mat int) {
	sy := ints.MinInt(st.Y, ed.Y)
	ey := ints.MaxInt(st.Y, ed.Y)
	for y := sy; y <= ey; y++ {
		ev.World.Set([]int{y, st.X}, mat)
	}
}

// WorldLine draw line in world with given mat
func (ev *FWorld) WorldLine(st, ed evec.Vec2i, mat int) {
	di := ed.Sub(st)

	if di.X == 0 {
		ev.WorldLineVert(st, ed, mat)
		return
	}
	if di.Y == 0 {
		ev.WorldLineHoriz(st, ed, mat)
		return
	}

	dv := di.ToVec2()
	dst := dv.Length()
	v := NormVecLine(dv)
	op := st.ToVec2()
	cp := op
	gp := evec.Vec2i{}
	for {
		cp, gp = NextVecPoint(cp, v)
		ev.SetWorld(gp, mat)
		d := cp.DistTo(op) // not very efficient, but works.
		if d >= dst {
			break
		}
	}
}

// WorldRandom distributes n of given material in random locations
func (ev *FWorld) WorldRandom(n, mat int) {
	cnt := 0
	for cnt < n {
		px := rand.Intn(ev.Size.X)
		py := rand.Intn(ev.Size.Y)
		ix := []int{py, px}
		cm := ev.World.Value(ix)
		if cm == 0 {
			ev.World.Set(ix, mat)
			cnt++
		}
	}
}

// WorldRect draw rectangle in world with given mat
func (ev *FWorld) WorldRect(st, ed evec.Vec2i, mat int) {
	ev.WorldLineHoriz(st, evec.Vec2i{ed.X, st.Y}, mat)
	ev.WorldLineHoriz(evec.Vec2i{st.X, ed.Y}, evec.Vec2i{ed.X, ed.Y}, mat)
	ev.WorldLineVert(st, evec.Vec2i{st.X, ed.Y}, mat)
	ev.WorldLineVert(evec.Vec2i{ed.X, st.Y}, evec.Vec2i{ed.X, ed.Y}, mat)
}

// GenWorld generates a world -- edit to create in way desired
func (ev *FWorld) GenWorld() {
	wall := ev.MatMap["Wall"]
	food := ev.MatMap["Food"]
	water := ev.MatMap["Water"]
	ev.World.SetZeros()
	// always start with a wall around the entire world -- no seeing the turtles..
	ev.WorldRect(evec.Vec2i{0, 0}, evec.Vec2i{ev.Size.X - 1, ev.Size.Y - 1}, wall)
	ev.WorldRect(evec.Vec2i{20, 20}, evec.Vec2i{40, 40}, wall)
	ev.WorldRect(evec.Vec2i{60, 60}, evec.Vec2i{80, 80}, wall)

	ev.WorldLine(evec.Vec2i{60, 20}, evec.Vec2i{80, 40}, wall) // double-thick lines = no leak
	ev.WorldLine(evec.Vec2i{60, 19}, evec.Vec2i{80, 39}, wall)

	// don't put anything in center starting point
	ctr := ev.Size.DivScalar(2)
	ev.SetWorld(ctr, wall)

	ev.WorldRandom(50, food)
	ev.WorldRandom(50, water)

	// clear center
	ev.SetWorld(ctr, 0)
}

////////////////////////////////////////////////////////////////////
// Subcortex / Instinct

// ActGenTrace prints trace of act gen if enabled
func (ev *FWorld) ActGenTrace(desc string, act int) {
	if !ev.TraceActGen {
		return
	}
	fmt.Printf("%s: act: %s\n", desc, ev.Acts[act])
}

// ActGen generates an action for current situation based on simple
// coded heuristics -- i.e., what subcortical evolutionary instincts provide.
// Also returns the urgency score as a probability -- if urgency is 1
// then the generated action should definitely be used.  The default is 0,
// which is the baseline.
func (ev *FWorld) ActGen() (int, float32) {
	wall := ev.MatMap["Wall"]
	food := ev.MatMap["Food"]
	water := ev.MatMap["Water"]
	left := ev.ActMap["Left"]
	right := ev.ActMap["Right"]
	eat := ev.ActMap["Eat"]

	nmat := len(ev.Mats)
	frmat := ints.MinInt(ev.ProxMats[0], nmat)

	// get info about what is in fovea
	fsz := 1 + 2*ev.FoveaSize
	fwt := float32(0)
	wwt := float32(0)
	fdp := float32(100000)
	wdp := float32(100000)
	fovdp := float32(100000)
	fovnonwall := 0
	for i := 0; i < fsz; i++ {
		mat := ev.FovMats[i]
		switch {
		case mat == water:
			wwt += 1 - ev.FovDepthLogs[i] // more weight if closer
			wdp = mat32.Min(wdp, ev.FovDepths[i])
		case mat == food:
			fwt += 1 - ev.FovDepthLogs[i] // more weight if closer
			fdp = mat32.Min(fdp, ev.FovDepths[i])
		case mat <= ev.BarrierIdx:
		default:
			fovnonwall = mat
		}
		fovdp = mat32.Min(fovdp, ev.FovDepths[i])
	}
	fwt *= 1 - ev.InterStates["Energy"] // weight by need
	wwt *= 1 - ev.InterStates["Hydra"]

	fovmat := ev.FovMats[ev.FoveaSize]
	fovmats := ev.Mats[fovmat]

	// get info about full depth view
	minl := 1.0
	minr := 1.0
	avgl := 0.0
	avgr := 0.0
	hang := ev.NFOVRays / 2
	for i := 0; i < ev.NFOVRays; i++ {
		dp := float64(ev.DepthLogs[i])
		if i < hang-1 {
			minl = math.Min(minl, dp)
			avgl += dp
		} else if i > hang+1 {
			minr = math.Min(minr, dp)
			avgr += dp
		}
	}
	ldf := 1 - minl
	rdf := 1 - minr
	if math.Abs(minl-minr) < 0.1 {
		ldf = 1 - (avgl / float64(hang-1))
		rdf = 1 - (avgr / float64(hang-1))
	}
	smaxpow := 10.0
	rlp := float64(.5)
	if ldf+rdf > 0 {
		rpow := math.Exp(rdf * smaxpow)
		lpow := math.Exp(ldf * smaxpow)
		rlp = float64(lpow / (rpow + lpow))
	}
	rlact := left // right or left
	if erand.BoolProb(rlp, -1) {
		rlact = right
	}
	// fmt.Printf("rlp: %.3g  ldf: %.3g  rdf: %.3g  act: %s\n", rlp, ldf, rdf, ev.Acts[rlact])
	rlps := fmt.Sprintf("%.3g", rlp)

	lastact := ev.Act
	frnd := rand.Float32()

	farDist := float32(10)
	farTurnP := float32(0.2)
	rndExpSame := float32(0.33)
	rndExpTurn := float32(0.33)

	urgency := float32(0)
	act := ev.ActMap["Forward"] // default
	switch {
	case frmat == wall:
		if lastact == left || lastact == right {
			act = lastact // keep going
			ev.ActGenTrace("at wall, keep turning", act)
		} else {
			act = rlact
			ev.ActGenTrace(fmt.Sprintf("at wall, rlp: %s, turn", rlps), act)
		}
		urgency = ev.WallUrgency
	case frmat == food:
		act = ev.ActMap["Eat"]
		ev.ActGenTrace("at food", act)
		urgency = ev.EatUrgency
	case frmat == water:
		act = ev.ActMap["Drink"]
		ev.ActGenTrace("at water", act)
		urgency = ev.EatUrgency
	case fwt > wwt: // food more than water
		wts := fmt.Sprintf("fwt: %g > wwt: %g, dist: %g", fwt, wwt, fdp)
		if fdp > farDist { // far away
			urgency = 0
			if frnd < farTurnP {
				act = rlact
				ev.ActGenTrace(fmt.Sprintf("far food in view (%s), explore, rlp: %s, turn", wts, rlps), act)
			} else {
				ev.ActGenTrace("far food in view "+wts, act)
			}
		} else {
			urgency = ev.CloseUrgency
			ev.ActGenTrace("close food in view "+wts, act)
		}
	case wwt > fwt: // water more than food
		wts := fmt.Sprintf("wwt: %g > fwt: %g, dist: %g", wwt, fwt, wdp)
		if wdp > farDist { // far away
			urgency = 0
			if frnd < farTurnP {
				act = rlact
				ev.ActGenTrace(fmt.Sprintf("far water in view (%s), explore, rlp: %s, turn", wts, rlps), act)
			} else {
				ev.ActGenTrace("far water in view "+wts, act)
			}
		} else {
			urgency = ev.CloseUrgency
			ev.ActGenTrace("close water in view "+wts, act)
		}
	case fovdp < 4 && fovnonwall == 0: // close to wall
		urgency = ev.CloseUrgency
		if lastact == left || lastact == right {
			act = lastact // keep going
			ev.ActGenTrace("close to: "+fovmats+" keep turning", act)
		} else {
			act = rlact
			ev.ActGenTrace(fmt.Sprintf("close to: %s rlp: %s, turn", fovmats, rlps), act)
		}
	default: // random explore -- nothing obvious
		urgency = 0
		switch {
		case frnd < rndExpSame && lastact < eat:
			act = lastact // continue
			ev.ActGenTrace("looking at: "+fovmats+" repeat last act", act)
		case frnd < rndExpSame+rndExpTurn:
			act = rlact
			ev.ActGenTrace("looking at: "+fovmats+" turn", act)
		default:
			ev.ActGenTrace("looking at: "+fovmats+" go", act)
		}
	}

	ev.Urgency = urgency
	return act, urgency
}

func (ev FWorld) PossiblyGetTeachingSignal(agentChosenAction int) (chosenAction int) {
	heuristicAction, _ := ev.ActGen() // TODO What does urgency do?
	//actionName := ev.Acts[heuristicAction]
	if heuristicAction == agentChosenAction {
		//ev.ActMatch = 1 // TODO Maybe record this for metrics reasons
	}
	// TODO Reduce PctCortex over time
	if rand.Float64() < ev.PctCortex {
		return agentChosenAction
	} else {
		return heuristicAction
	}
}

// GetActionIdFromVL decodes a tensor from VL.
func (ev FWorld) GetActionIdFromVL(actions map[string]agent.Action) (actionId int) {
	vt := etensor.Float32{} // TODO This data type conversion should be cleaned up
	val, ok := actions["VL"]
	if ok {
		vt.Values = make([]float32, val.Vector.Len())
		vt.CopyFrom(val.Vector)
		agentActionId := ev.DecodeAct(vt)
		return agentActionId
	} else {
		return 0
	}
}

// ApplyTeachingFunction gets the teaching signal.
func (ev FWorld) ApplyTeachingFunction(agentActionId int) int {
	chosenAction := ev.PossiblyGetTeachingSignal(agentActionId)
	agentAction := ev.Acts[agentActionId]
	chosenActionStr := ev.Acts[chosenAction]
	log.Info("Action from Agent: " + agentAction + ", but Heuristic Advises: " + chosenActionStr)
	return chosenAction
}

func recordPerformance(ev *FWorld, chosenAction int) {
	bestAction, _ := ev.ActGen()
	ev.bestActions = append(ev.bestActions, int32(bestAction))
	ev.predictedActions = append(ev.predictedActions, int32(chosenAction))
}

// StepWorld looks at the action vector and converts it into an actual action that it takes in the world.
func (ev *FWorld) StepWorld(chosenAction int, agentDone bool) (done bool, debug string) {
	chosenActionStr := ev.Acts[chosenAction]

	log.Info("Taking action: " + chosenActionStr)
	ev.Action(chosenActionStr, nil)
	ev.Step()
	if ev.UseGUI {
		ev.UpdateWorldGui()
	}
	return false, ""
}

func (ev FWorld) calculateAndRecordReward(obs *map[string]*net_env.ETensor) {
	// TODO Maybe treat Hydra and Energy differently since they're temporally discounted or something.
	reward := ev.InterStates["FoodRew"] + ev.InterStates["WaterRew"] + ev.InterStates["Energy"] + ev.InterStates["Hydra"] - ev.InterStates["BumpPain"]

	(*obs)["Reward"] = &net_env.ETensor{Shape: &net_env.Shape{
		Shape:  []int32{1},
		Stride: []int32{1},
		Names:  []string{"Reward"},
	}, Values: []float64{float64(reward)}}
}

func (ev FWorld) intToETensor(action int, name string) *net_env.ETensor {
	return ev.floatToETensor(float64(action), name)
}

func (ev FWorld) floatToETensor(action float64, name string) *net_env.ETensor {
	return &net_env.ETensor{Shape: &net_env.Shape{
		Shape:  []int32{1},
		Stride: []int32{1},
		Names:  []string{name},
	}, Values: []float64{float64(action)}}
}

func (ev *FWorld) getAllObservations() map[string]*net_env.ETensor {
	obs := map[string]*net_env.ETensor{}
	// These two lists line up which network layer corresponds to which state within FWorld.
	states := []string{"Depth", "FovDepth", "Fovea", "ProxSoma", "Vestibular", "Inters", "Action", "Action"}
	layers := []string{"V2Wd", "V2Fd", "V1F", "S1S", "S1V", "Ins", "VL", "Act"}
	for i, lnm := range layers {
		obs[lnm] = network_agent.FromTensor(ev.State(states[i]))
	}

	//specific to the VL target, use the expected output
	expectedAction, _ := ev.ActGen()
	genAction := ev.Acts[expectedAction]
	actionTensor := ev.Pats[genAction]
	obs["VL"] = network_agent.FromTensor(actionTensor)            //Heuristic is a distributed rep of best
	obs["Heuristic"] = ev.intToETensor(expectedAction, genAction) //This is a a discrete action of best action
	obs["PredictedActionLastTimeStep"] = ev.intToETensor(ev.LastActionPredicted, "PredictedActionLastTimeStep")
	ev.calculateAndRecordReward(&obs) // Add reward.
	//todo these shouldn't be called every step, perhaps ever %ticks, still minor but will be basically M(steps) N (size) operation
	obs["F1"] = ev.floatToETensor(metrics.F1ScoreMacro(ev.predictedActions, ev.bestActions, []int32{0, 1, 2, 3, 4}), "F1Score") // Add F1Score.
	obs["KL"] = ev.floatToETensor(metrics.KLDivergence(ev.predictedActions, ev.bestActions), "KL")                              // Add KL.
	return obs
}

// TODO(Erdal) These helper methods are being moved to a central location. WIP by Erdal.

func toInt(xs []int32) []int {
	if xs == nil {
		return nil
	}
	res := make([]int, len(xs))
	for i := range xs {
		res[i] = int(xs[i])
	}
	return res
}

func toTensor(envtensor *net_env.ETensor) etensor.Tensor {
	return etensor.NewFloat64Shape(toShape(envtensor.Shape), envtensor.Values)
}

func toShape(shape *net_env.Shape) *etensor.Shape {
	return etensor.NewShape(toInt(shape.Shape), toInt(shape.Stride), shape.Names)
}

func transformActions(actions net_env.Actions) map[string]agent.Action {
	res := map[string]agent.Action{}
	for k, v := range actions {
		res[k] = agent.Action{
			ActionShape: &agent.SpaceSpec{
				ContinuousShape: toShape(v.ActionShape.Shape).Shp,
				Stride:          toShape(v.ActionShape.Shape).Strd,
				Min:             v.ActionShape.Min,
				Max:             v.ActionShape.Max,
			},
			Vector: toTensor(v.Vector),
		}
	}
	return res
}

////////////////////////////////
// GUI STUFF

func (ev *FWorld) UpdateWorldGui() {
	if ev.Scene.Chg { // something important happened, refresh
		ev.Trace.CopyFrom(ev.World)
	}

	nc := len(ev.Mats)
	ev.Trace.Set([]int{ev.PosI.Y, ev.PosI.X}, nc+ev.Angle/ev.AngInc)

	updt := ev.WorldTabs.UpdateStart()
	ev.TraceView.UpdateSig()
	ev.WorldTabs.UpdateEnd(updt)
}

func (ev *FWorld) Left() {
	ev.Action("Left", nil)
	ev.UpdateWorldGui()
}

func (ev *FWorld) Right() {
	ev.Action("Right", nil)
	ev.UpdateWorldGui()
}

func (ev *FWorld) Forward() {
	ev.Action("Forward", nil)
	ev.UpdateWorldGui()
}

func (ev *FWorld) Backward() {
	ev.Action("Backward", nil)
	ev.UpdateWorldGui()
}

func (ev *FWorld) Eat() {
	ev.Action("Eat", nil)
	ev.UpdateWorldGui()
}

func (ev *FWorld) Drink() {
	ev.Action("Drink", nil)
	ev.UpdateWorldGui()
}

// ConfigWorldGui configures all the world view GUI elements
func (ev *FWorld) ConfigWorldGui() *gi.Window {

	// order: Empty, wall, food, water, foodwas, waterwas
	ev.MatColors = []string{"lightgrey", "black", "orange", "blue", "brown", "navy"}

	ev.Trace = ev.World.Clone().(*etensor.Int)

	width := 1600
	height := 1200

	win := gi.NewMainWindow("fworld", "Flat World", width, height)
	ev.WorldWin = win

	vp := win.WinViewport2D()
	updt := vp.UpdateStart()

	mfr := win.SetMainFrame()

	toolbar := gi.AddNewToolBar(mfr, "tbar")
	toolbar.SetStretchMaxWidth()

	split := gi.AddNewSplitView(mfr, "split")
	split.Dim = mat32.X
	split.SetStretchMax()

	sv := giv.AddNewStructView(split, "sv")
	sv.SetStruct(&ev)

	tv := gi.AddNewTabView(split, "tv")
	ev.WorldTabs = tv

	tg := tv.AddNewTab(etview.KiT_TensorGrid, "Trace").(*etview.TensorGrid)
	ev.TraceView = tg
	tg.SetTensor(ev.Trace)
	ev.ConfigWorldView(tg)

	wg := tv.AddNewTab(etview.KiT_TensorGrid, "World").(*etview.TensorGrid)
	ev.WorldView = wg
	wg.SetTensor(ev.World)
	ev.ConfigWorldView(wg)

	split.SetSplits(.3, .7)

	toolbar.AddAction(gi.ActOpts{Label: "Init", Icon: "reset", Tooltip: "Init env.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ev.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ev.Init(0)
		vp.SetFullReRender()
	})

	toolbar.AddAction(gi.ActOpts{Label: "Left", Icon: "wedge-left", Tooltip: "Rotate Left", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ev.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ev.Left()
		vp.SetFullReRender()
	})

	toolbar.AddAction(gi.ActOpts{Label: "Right", Icon: "wedge-right", Tooltip: "Rotate Right", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ev.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ev.Right()
		vp.SetFullReRender()
	})

	toolbar.AddAction(gi.ActOpts{Label: "Forward", Icon: "wedge-up", Tooltip: "Step Forward", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ev.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ev.Forward()
		vp.SetFullReRender()
	})

	toolbar.AddAction(gi.ActOpts{Label: "Backward", Icon: "wedge-down", Tooltip: "Step Backward", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ev.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ev.Backward()
		vp.SetFullReRender()
	})

	toolbar.AddSeparator("sep-eat")

	toolbar.AddAction(gi.ActOpts{Label: "Eat", Icon: "field", Tooltip: "Eat food -- only if directly in front", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ev.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ev.Eat()
		vp.SetFullReRender()
	})

	toolbar.AddAction(gi.ActOpts{Label: "Drink", Icon: "svg", Tooltip: "Drink water -- only if directly in front", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ev.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ev.Drink()
		vp.SetFullReRender()
	})

	alreadyConnected := false // Don't let the user connect twice
	toolbar.AddAction(gi.ActOpts{Label: "Connect to Server and Run", Icon: "svg", Tooltip: "Connect to an intelligent model that is serving actions as a server.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ev.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ev.paused = false
		if !alreadyConnected {
			go connectAndQueryAgent(ev)
			alreadyConnected = true
		}
	})

	toolbar.AddAction(gi.ActOpts{Label: "Pause", Icon: "pause", Tooltip: "Pause stepping", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ev.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ev.paused = true
	})

	toolbar.AddAction(gi.ActOpts{Label: "Step 1", Icon: "step", Tooltip: "Step one timestep", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ev.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ev.paused = false
		ev.stepOne = true
	})

	toolbar.AddAction(gi.ActOpts{Label: "Run", Icon: "svg", Tooltip: "Run continuously forever", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ev.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ev.paused = false
		ev.stepOne = false
	})

	vp.UpdateEndNoSig(updt)

	// main menu
	appnm := gi.AppName()
	mmen := win.MainMenu
	mmen.ConfigMenus([]string{appnm, "File", "Edit", "Window"})

	amen := win.MainMenu.ChildByName(appnm, 0).(*gi.Action)
	amen.Menu.AddAppMenu(win)

	emen := win.MainMenu.ChildByName("Edit", 1).(*gi.Action)
	emen.Menu.AddCopyCutPaste(win)

	win.MainMenuUpdated()
	return win
}

func (ev *FWorld) ConfigWorldView(tg *etview.TensorGrid) {
	cnm := "FWorldColors"
	cm, ok := colormap.AvailMaps[cnm]
	if !ok {
		cm = &colormap.Map{}
		cm.Name = cnm
		cm.Indexed = true
		nc := len(ev.Mats)
		cm.Colors = make([]gist.Color, nc+ev.NRotAngles)
		cm.NoColor = gist.Black
		for i, cnm := range ev.MatColors {
			cm.Colors[i].SetString(cnm, nil)
		}
		ch := colormap.AvailMaps["ColdHot"]
		for i := 0; i < ev.NRotAngles; i++ {
			nv := float64(i) / float64(ev.NRotAngles-1)
			cm.Colors[nc+i] = ch.Map(nv) // color map of rotation
		}
		colormap.AvailMaps[cnm] = cm
	}
	tg.Disp.Defaults()
	tg.Disp.ColorMap = giv.ColorMapName(cnm)
	tg.Disp.GridFill = 1
	tg.SetStretchMax()
}

func stepWorldAndAgentOnce(ev *FWorld, agent *net_env.AgentClient, defaultCtx context.Context) (success bool) {
	observations := ev.getAllObservations()
	// Step the agent
	actions, _ := agent.Step(defaultCtx, observations, "episode:"+strconv.Itoa(ev.Tick.Cur))
	if actions == nil {
		log.Error("disconnected from agent, exiting")
		return false
	}

	move, ok := actions["move"] // This contains a discrete option.

	//todo this shouldn't be a switch, we need a parasiminous way of handling model <-> env input out shapes
	if ok {
		// We've received a discrete action and can use it directly to StepWorld
		ev.StepWorld(int(move.DiscreteOption), false)
		recordPerformance(ev, int(move.DiscreteOption))
	} else {
		// Assume VL is in actions and treat it continuously and also apply the teaching function.
		modelGeneratedAction := ev.GetActionIdFromVL(transformActions(actions))
		recordPerformance(ev, int(modelGeneratedAction))
		ev.LastActionPredicted = modelGeneratedAction //action suggested for the previous time step
		ev.StepWorld(ev.ApplyTeachingFunction(modelGeneratedAction), false)
	}
	return true
}

func connectAndQueryAgent(ev *FWorld) {
	agent := network.MakeClient()
	var defaultCtx = context.Background()

	default_action, _ := agent.Init(defaultCtx, nil, nil)
	log.Info("Initializing FWorld") // TODO Delete
	if default_action == nil {
		log.Error("failed to connect")
	} else { //if successfully connected
		// This is the main loop.
		for {
			if ev.paused {
				continue
			}
			if !stepWorldAndAgentOnce(ev, agent, defaultCtx) {
				break
			}
			if ev.stepOne {
				ev.paused = true
				ev.stepOne = false
			}
		}
	}
}

var gConfig Config

func main() {
	gConfig.Load() // LATER specify the .cfg as a cmd line arg

	err := log.Init(
		log.SetLevel(log.INFO),
		log.SetFileName("world.log"),
	)
	if err != nil {
		panic(err)
	}

	bestWorld := FWorld{}
	bestWorld.UseGUI = gConfig.GUI
	bestWorld.Config(200)
	bestWorld.InitWorld(nil)

	if !bestWorld.UseGUI {
		connectAndQueryAgent(&bestWorld)
	} else {

		gimain.Main(func() {
			fwin := bestWorld.ConfigWorldGui()
			// TODO Get the server connection to start automatically on window open

			fwin.StartEventLoop()
		})
	}
}

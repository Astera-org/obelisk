// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/json"
	"fmt"
	"image"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"sort"

	"github.com/emer/emergent/env"
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/evec"
	"github.com/emer/emergent/patgen"
	"github.com/emer/empi/empi"
	"github.com/emer/empi/mpi"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/metric"
	"github.com/emer/etable/minmax"
	"github.com/goki/gi/gi"
	"github.com/goki/ki/ints"
	"github.com/goki/mat32"
	"golang.org/x/image/draw"
	"golang.org/x/image/math/f64"
)

// ImagesEnv provides the rendered results of the Obj3D + Saccade generator.
type ImagesEnv struct {
	Nm         string       `desc:"name of this environment"`
	Dsc        string       `desc:"description of this environment"`
	ImageFile  string       `desc:"image file name"`
	Test       bool         `desc:"present test items, else train"`
	Sequential bool         `desc:"present items in sequential order -- else shuffled"`
	High16     bool         `desc:"compute high-res full field filtering"`
	ColorDoG   bool         `desc:"compute color DoG (blob) filtering"`
	Images     Images       `desc:"images list"`
	TransMax   mat32.Vec2   `desc:"def 0.3 maximum amount of translation as proportion of half-width size in each direction -- 1 = something in center is now at right edge"`
	TransSigma float32      `def:"0.15" desc:"if > 0, generate translations using gaussian normal distribution with this standard deviation, and then clip to TransMax range -- this facilitates learning on the central region while still giving exposure to wider area.  Tyically turn off for last 100 epochs to measure true uniform distribution performance."`
	ScaleRange minmax.F32   `desc:"def 0.5 - 1.1 range of scale"`
	RotateMax  float32      `def:"8" desc:"def 8 maximum degrees of rotation in plane -- image is rotated plus or minus in this range"`
	Img        V1Img        `desc:"image that we operate upon -- one image shared among all filters"`
	V1l16      Vis          `desc:"v1 16deg low resolution filtering of image -- V1AllTsr has result"`
	V1m16      Vis          `desc:"v1 16deg medium resolution filtering of image -- V1AllTsr has result"`
	V1h16      Vis          `desc:"v1 16deg high resolution filtering of image -- V1AllTsr has result"`
	V1l8       Vis          `desc:"v1 8deg low resolution filtering of image -- V1AllTsr has result"`
	V1m8       Vis          `desc:"v1 8deg medium resolution filtering of image -- V1AllTsr has result"`
	V1Cl16     ColorVis     `desc:"v1 color 16deg low resolution filtering of image -- OutAll has result"`
	V1Cm16     ColorVis     `desc:"v1 color 16deg medium resolution filtering of image -- OutAll has result"`
	V1Cl8      ColorVis     `desc:"v1 color 8deg low resolution filtering of image -- OutAll has result"`
	V1Cm8      ColorVis     `desc:"v1 color 8deg medium resolution filtering of image -- OutAll has result"`
	MaxOut     int          `desc:"maximum number of output categories representable here"`
	OutRandom  bool         `desc:"use random bit patterns instead of localist output units"`
	RndPctOn   float32      `desc:"proportion activity for random patterns"`
	RndMinDiff float32      `desc:"proportion minimum difference for random patterns"`
	OutSize    evec.Vec2i   `desc:"the output tensor geometry -- must be >= number of cats"`
	NOutPer    int          `desc:"number of output units per category -- spiking may benefit from replication -- is Y inner dim of output tensor"`
	Pats       etable.Table `view:"no-inline" desc:"output patterns: either localist or random"`

	Output    etensor.Float32 `desc:"output pattern for current item"`
	StRow     int             `desc:"starting row, e.g., for mpi allocation across processors"`
	EdRow     int             `desc:"ending row -- if 0 it is ignored"`
	Shuffle   []int           `desc:"suffled list of entire set of images -- re-shuffle every time through imgidxs"`
	ImgIdxs   []int           `desc:"indexs of images to present -- from StRow to EdRow"`
	Run       env.Ctr         `view:"inline" desc:"current run of model as provided during Init"`
	Epoch     env.Ctr         `view:"inline" desc:"arbitrary aggregation of trials, for stats etc"`
	Trial     env.Ctr         `view:"inline" desc:"each object trajectory is one trial"`
	Row       env.Ctr         `view:"inline" desc:"row of item list  -- this is actual counter driving everything"`
	CurCat    string          `desc:"current category"`
	CurCatIdx int             `desc:"index of current category"`
	CurImg    string          `desc:"current image"`
	CurTrans  mat32.Vec2      `desc:"current translation"`
	CurScale  float32         `desc:"current scaling"`
	CurRot    float32         `desc:"current rotation"`

	Image image.Image `view:"-" desc:"rendered image as loaded"`
}

func (ev *ImagesEnv) Name() string { return ev.Nm }
func (ev *ImagesEnv) Desc() string { return ev.Dsc }

func (ev *ImagesEnv) Validate() error {
	return nil
}

func (ev *ImagesEnv) Defaults() {
	ev.TransSigma = 0
	ev.TransMax.Set(0.2, 0.2)
	ev.ScaleRange.Set(0.8, 1.1)
	ev.RotateMax = 8
	ev.RndPctOn = 0.2
	ev.RndMinDiff = 0.5
	ev.NOutPer = 5
	ev.Img.Defaults()
	ev.V1l16.Defaults(0, 24, 8, &ev.Img)
	ev.V1m16.Defaults(0, 12, 4, &ev.Img)
	ev.V1h16.Defaults(0, 6, 2, &ev.Img)
	ev.V1l8.Defaults(32, 12, 4, &ev.Img)
	ev.V1m8.Defaults(32, 6, 2, &ev.Img)

	ev.V1Cl16.Defaults(0, 16, 16, &ev.Img)
	ev.V1Cm16.Defaults(0, 8, 8, &ev.Img)
	ev.V1Cl8.Defaults(32, 8, 8, &ev.Img)
	ev.V1Cm8.Defaults(32, 4, 4, &ev.Img)
}

// ImageList returns the list of images -- train or test
func (ev *ImagesEnv) ImageList() []string {
	if ev.Test {
		return ev.Images.FlatTest
	}
	return ev.Images.FlatTrain
}

// MPIAlloc allocate objects based on mpi processor number
func (ev *ImagesEnv) MPIAlloc() {
	ws := mpi.WorldSize()
	nim := ws * (len(ev.ImageList()) / ws) // even multiple of size -- few at end are lost..
	ev.StRow, ev.EdRow, _ = empi.AllocN(nim)
	// mpi.PrintAllProcs = true
	// mpi.Printf("allocated images: n: %d st: %d ed: %d\n", nim, ev.StRow, ev.EdRow)
	// mpi.PrintAllProcs = false
}

func (ev *ImagesEnv) Init(run int) {
	ev.Run.Scale = env.Run
	ev.Epoch.Scale = env.Epoch
	ev.Trial.Scale = env.Trial
	ev.Row.Scale = env.Tick
	ev.Run.Init()
	ev.Epoch.Init()
	ev.Trial.Init()
	ev.Run.Cur = run
	ev.Row.Cur = -1 // init state -- key so that first Step() = 0
	nitm := len(ev.ImageList())
	if ev.EdRow > 0 {
		ev.EdRow = ints.MinInt(ev.EdRow, nitm)
		ev.ImgIdxs = make([]int, ev.EdRow-ev.StRow)
	} else {
		ev.ImgIdxs = make([]int, nitm)
	}
	for i := range ev.ImgIdxs {
		ev.ImgIdxs[i] = ev.StRow + i
	}
	ev.Shuffle = rand.Perm(nitm)
	ev.Row.Max = len(ev.ImgIdxs)
	nc := len(ev.Images.Cats)
	ev.MaxOut = ints.MaxInt(nc, ev.MaxOut)
	ev.ConfigPats()
}

// SaveListJSON saves flat string list to a JSON-formatted file.
func SaveListJSON(list []string, filename string) error {
	b, err := json.MarshalIndent(list, "", "  ")
	if err != nil {
		log.Println(err) // unlikely
		return err
	}
	err = ioutil.WriteFile(string(filename), b, 0644)
	if err != nil {
		log.Println(err)
	}
	return err
}

// OpenListJSON opens flat string list from a JSON-formatted file.
func OpenListJSON(list *[]string, filename string) error {
	b, err := ioutil.ReadFile(string(filename))
	if err != nil {
		log.Println(err)
		return err
	}
	return json.Unmarshal(b, list)
}

// SaveList2JSON saves double-string list to a JSON-formatted file.
func SaveList2JSON(list [][]string, filename string) error {
	b, err := json.MarshalIndent(list, "", "  ")
	if err != nil {
		log.Println(err) // unlikely
		return err
	}
	err = ioutil.WriteFile(string(filename), b, 0644)
	if err != nil {
		log.Println(err)
	}
	return err
}

// OpenList2JSON opens double-string list from a JSON-formatted file.
func OpenList2JSON(list *[][]string, filename string) error {
	b, err := ioutil.ReadFile(string(filename))
	if err != nil {
		log.Println(err)
		return err
	}
	return json.Unmarshal(b, list)
}

// OpenConfig opens saved configuration for current images
func (ev *ImagesEnv) OpenConfig() bool {
	cfnm := fmt.Sprintf("%s_cats.json", ev.ImageFile)
	tsfnm := fmt.Sprintf("%s_ntest%d_tst.json", ev.ImageFile, ev.Images.NTestPerCat)
	trfnm := fmt.Sprintf("%s_ntest%d_trn.json", ev.ImageFile, ev.Images.NTestPerCat)
	_, err := os.Stat(tsfnm)
	if !os.IsNotExist(err) {
		OpenListJSON(&ev.Images.Cats, cfnm)
		OpenList2JSON(&ev.Images.ImagesTest, tsfnm)
		OpenList2JSON(&ev.Images.ImagesTrain, trfnm)
		ev.Images.ToTrainAll()
		ev.Images.Flats()
		return true
	}
	return false
}

// SaveConfig saves configuration for current images
func (ev *ImagesEnv) SaveConfig() {
	cfnm := fmt.Sprintf("%s_cats.json", ev.ImageFile)
	tsfnm := fmt.Sprintf("%s_ntest%d_tst.json", ev.ImageFile, ev.Images.NTestPerCat)
	trfnm := fmt.Sprintf("%s_ntest%d_trn.json", ev.ImageFile, ev.Images.NTestPerCat)
	SaveListJSON(ev.Images.Cats, cfnm)
	SaveList2JSON(ev.Images.ImagesTest, tsfnm)
	SaveList2JSON(ev.Images.ImagesTrain, trfnm)
}

// ConfigPats configures the output patterns
func (ev *ImagesEnv) ConfigPats() {
	if ev.OutRandom {
		ev.ConfigPatsRandom()
	} else {
		ev.ConfigPatsLocalist()
	}
}

// ConfigPatsName names the patterns
func (ev *ImagesEnv) ConfigPatsName() {
	for i := 0; i < ev.MaxOut; i++ {
		nm := fmt.Sprintf("P%03d", i)
		if i < len(ev.Images.Cats) {
			nm = ev.Images.Cats[i]
		}
		ev.Pats.SetCellString("Name", i, nm)
	}
}

// ConfigPatsLocalist configures the output patterns: localist case
func (ev *ImagesEnv) ConfigPatsLocalist() {
	oshp := []int{ev.OutSize.Y, ev.OutSize.X, ev.NOutPer, 1}
	oshpnm := []string{"Y", "X", "NPer", "1"}
	ev.Output.SetShape(oshp, nil, oshpnm)
	sch := etable.Schema{
		{"Name", etensor.STRING, nil, nil},
		{"Output", etensor.FLOAT32, oshp, oshpnm},
	}
	ev.Pats.SetFromSchema(sch, ev.MaxOut)
	for pi := 0; pi < ev.MaxOut; pi++ {
		out := ev.Pats.CellTensor("Output", pi)
		si := ev.NOutPer * pi
		for i := 0; i < ev.NOutPer; i++ {
			out.SetFloat1D(si+i, 1)
		}
	}
	ev.ConfigPatsName()
}

// ConfigPatsRandom configures the output patterns: random case
func (ev *ImagesEnv) ConfigPatsRandom() {
	oshp := []int{ev.OutSize.Y, ev.OutSize.X}
	oshpnm := []string{"Y", "X"}
	ev.Output.SetShape(oshp, nil, oshpnm)
	sch := etable.Schema{
		{"Name", etensor.STRING, nil, nil},
		{"Output", etensor.FLOAT32, oshp, oshpnm},
	}
	ev.Pats.SetFromSchema(sch, ev.MaxOut)
	np := ev.OutSize.X * ev.OutSize.Y
	nOn := patgen.NFmPct(ev.RndPctOn, np)
	minDiff := patgen.NFmPct(ev.RndMinDiff, nOn)
	fnm := fmt.Sprintf("rndpats_%dx%d_n%d_on%d_df%d.tsv", ev.OutSize.X, ev.OutSize.Y, ev.MaxOut, nOn, minDiff)
	_, err := os.Stat(fnm)
	if !os.IsNotExist(err) {
		ev.Pats.OpenCSV(gi.FileName(fnm), etable.Tab)
	} else {
		out := ev.Pats.Col(1).(*etensor.Float32)
		patgen.PermutedBinaryMinDiff(out, nOn, 1, 0, minDiff)
		ev.ConfigPatsName()
		ev.Pats.SaveCSV(gi.FileName(fnm), etable.Tab, etable.Headers)
	}
}

// NewShuffle generates a new random order of items to present
func (ev *ImagesEnv) NewShuffle() {
	erand.PermuteInts(ev.Shuffle)
}

// CurImage returns current image based on row and
func (ev *ImagesEnv) CurImage() string {
	il := ev.ImageList()
	sz := len(ev.ImgIdxs)
	if ev.Row.Cur >= sz {
		ev.Row.Max = sz
		ev.Row.Cur = 0
		ev.NewShuffle()
	}
	r := ev.Row.Cur
	if r < 0 {
		r = 0
	}
	i := ev.ImgIdxs[r]
	if !ev.Sequential {
		i = ev.Shuffle[i]
	}
	ev.CurImg = il[i]
	ev.CurCat = ev.Images.Cat(ev.CurImg)
	ev.CurCatIdx = ev.Images.CatMap[ev.CurCat]
	return ev.CurImg
}

// OpenImage opens current image
func (ev *ImagesEnv) OpenImage() error {
	img := ev.CurImage()
	fnm := filepath.Join(ev.Images.Path, img)
	var err error
	ev.Image, err = gi.OpenImage(fnm)
	if err != nil {
		log.Println(err)
	}
	return err
}

// RandTransforms generates random transforms
func (ev *ImagesEnv) RandTransforms() {
	if ev.TransSigma > 0 {
		ev.CurTrans.X = float32(erand.Gauss(float64(ev.TransSigma), -1))
		ev.CurTrans.X = mat32.Clamp(ev.CurTrans.X, -ev.TransMax.X, ev.TransMax.X)
		ev.CurTrans.Y = float32(erand.Gauss(float64(ev.TransSigma), -1))
		ev.CurTrans.Y = mat32.Clamp(ev.CurTrans.Y, -ev.TransMax.Y, ev.TransMax.Y)
	} else {
		ev.CurTrans.X = (rand.Float32()*2 - 1) * ev.TransMax.X
		ev.CurTrans.Y = (rand.Float32()*2 - 1) * ev.TransMax.Y
	}
	ev.CurScale = ev.ScaleRange.Min + ev.ScaleRange.Range()*rand.Float32()
	ev.CurRot = (rand.Float32()*2 - 1) * ev.RotateMax
}

// TransformImage transforms the image according to current translation and scaling
func (ev *ImagesEnv) TransformImage() {
	s := mat32.NewVec2FmPoint(ev.Image.Bounds().Size())
	transformer := draw.BiLinear
	tx := 0.5 * ev.CurTrans.X * s.X
	ty := 0.5 * ev.CurTrans.Y * s.Y
	m := mat32.Translate2D(s.X*.5+tx, s.Y*.5+ty).Scale(ev.CurScale, ev.CurScale).Rotate(mat32.DegToRad(ev.CurRot)).Translate(-s.X*.5, -s.Y*.5)
	s2d := f64.Aff3{float64(m.XX), float64(m.XY), float64(m.X0), float64(m.YX), float64(m.YY), float64(m.Y0)}

	// use first color in upper left as fill color
	clr := ev.Image.At(0, 0)
	dst := image.NewRGBA(ev.Image.Bounds())
	src := image.NewUniform(clr)
	draw.Draw(dst, dst.Bounds(), src, image.ZP, draw.Src)

	transformer.Transform(dst, s2d, ev.Image, ev.Image.Bounds(), draw.Over, nil) // Over superimposes over bg
	ev.Image = dst
}

// FilterImage opens and filters current image
func (ev *ImagesEnv) FilterImage() error {
	err := ev.OpenImage()
	if err != nil {
		fmt.Println(err)
		return err
	}
	ev.TransformImage()
	ev.Img.SetImage(ev.Image, ev.V1l16.V1sGeom.FiltRt.X)
	ev.V1l16.Filter()
	ev.V1m16.Filter()
	ev.V1l8.Filter()
	ev.V1m8.Filter()
	if ev.High16 {
		ev.V1h16.Filter()
	}
	if ev.ColorDoG {
		ev.V1Cl16.Filter()
		ev.V1Cm16.Filter()
		ev.V1Cl8.Filter()
		ev.V1Cm8.Filter()
	}
	return nil
}

// SetOutput sets output by category
func (ev *ImagesEnv) SetOutput(out int) {
	ev.Output.SetZeros()
	ot := ev.Pats.CellTensor("Output", out)
	ev.Output.CopyCellsFrom(ot, 0, 0, ev.Output.Len())
}

// FloatIdx32 contains a float32 value and its index
type FloatIdx32 struct {
	Val float32
	Idx int
}

// ClosestRows32 returns the sorted list of distances from probe pattern
// and patterns in an etensor.Float32 where the outer-most dimension is
// assumed to be a row (e.g., as a column in an etable), using the given metric function,
// *which must have the Increasing property* -- i.e., larger = further.
// Col cell sizes must match size of probe (panics if not).
func ClosestRows32(probe *etensor.Float32, col *etensor.Float32, mfun metric.Func32) []FloatIdx32 {
	rows := col.Dim(0)
	csz := col.Len() / rows
	if csz != probe.Len() {
		panic("metric.ClosestRows32: probe size != cell size of tensor column!\n")
	}
	dsts := make([]FloatIdx32, rows)
	for ri := 0; ri < rows; ri++ {
		st := ri * csz
		rvals := col.Values[st : st+csz]
		v := mfun(probe.Values, rvals)
		dsts[ri].Val = v
		dsts[ri].Idx = ri
	}
	sort.Slice(dsts, func(i, j int) bool {
		return dsts[i].Val < dsts[j].Val
	})
	return dsts
}

// OutErr scores the output activity of network, returning the index of
// item with closest fit to given pattern, and 1 if that is error, 0 if correct.
// also returns a top-two error: if 2nd closest pattern was correct.
func (ev *ImagesEnv) OutErr(tsr *etensor.Float32) (maxi int, err, err2 float64) {
	ocol := ev.Pats.ColByName("Output").(*etensor.Float32)
	dsts := ClosestRows32(tsr, ocol, metric.InvCorrelation32)
	maxi = dsts[0].Idx
	err = 1.0
	if maxi == ev.CurCatIdx {
		err = 0
	}
	err2 = err
	if dsts[1].Idx == ev.CurCatIdx {
		err2 = 0
	}
	return
}

func (ev *ImagesEnv) String() string {
	return fmt.Sprintf("%s:%s_%d", ev.CurCat, ev.CurImg, ev.Trial.Cur)
}

func (ev *ImagesEnv) Step() bool {
	ev.Epoch.Same() // good idea to just reset all non-inner-most counters at start
	if ev.Row.Incr() {
		ev.NewShuffle()
	}
	if ev.Trial.Incr() {
		ev.Epoch.Incr()
	}
	ev.RandTransforms()
	ev.FilterImage()
	ev.SetOutput(ev.CurCatIdx)
	return true
}

func (ev *ImagesEnv) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
	switch scale {
	case env.Run:
		return ev.Run.Query()
	case env.Epoch:
		return ev.Epoch.Query()
	case env.Trial:
		return ev.Trial.Query()
	}
	return -1, -1, false
}

func (ev *ImagesEnv) State(element string) etensor.Tensor {
	switch element {
	case "V1l16":
		return &ev.V1l16.V1AllTsr
	case "V1m16":
		return &ev.V1m16.V1AllTsr
	case "V1h16":
		return &ev.V1h16.V1AllTsr
	case "V1l8":
		return &ev.V1l8.V1AllTsr
	case "V1m8":
		return &ev.V1m8.V1AllTsr
	case "V1Cl16":
		return &ev.V1Cl16.KwtaTsr
	case "V1Cm16":
		return &ev.V1Cm16.KwtaTsr
	case "V1Cl8":
		return &ev.V1Cl8.KwtaTsr
	case "V1Cm8":
		return &ev.V1Cm8.KwtaTsr
	case "Output":
		return &ev.Output
	}
	return nil
}

func (ev *ImagesEnv) Action(element string, input etensor.Tensor) {
	// nop
}

// Compile-time check that implements Env interface
var _ env.Env = (*ImagesEnv)(nil)

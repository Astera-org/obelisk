package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"github.com/emer/emergent/evec"
	"github.com/emer/emergent/popcode"
	"github.com/emer/etable/etensor"
	"github.com/goki/gi/gi"
	"github.com/goki/mat32"
	"io/ioutil"
	"os"
)

// TODO: add agent hunger and stuff (inters)

// this is actually the world from the agent's perspective
type World struct {
	Pixels         *etensor.Float64
	AgentX, AgentY int
	PosI           evec.Vec2i `inactive:"+" desc:"current location of agent, integer"`
	PosF           mat32.Vec2 `inactive:"+" desc:"current location of agent, floating point"`
	WorldX, WorldY int
	Size           evec.Vec2i `desc:"size of 2D world"`
	// TODO: we need to send this over from egan probably
	Angle int `inactive:"+" desc:"current angle, in degrees"`

	PatSize    evec.Vec2i
	Pats       map[string]*etensor.Float32
	Mats       []string
	AngInc     int
	FOV        int
	NFOVRays   int
	DepthSize  int
	DepthPools int
	DepthCode  popcode.OneD
	FoveaSize  int
	Dv         *etensor.Float32
	Dvr        *etensor.Float32
	Fd         *etensor.Float32
	Fdr        *etensor.Float32
	Fv         *etensor.Float32
}

func (w *World) getPoint(x, y int) (pixel uint32, err error) {
	// check for out of bounds first
	if x < 0 || y < 0 || x >= w.WorldY || y >= w.WorldY {
		return 0, errors.New("out of bounds")
	}
	// the number represents RGBA in that order
	return uint32(w.Pixels.At(x, y)), nil
}

// getMaterial finds the closest material based on a given pixel
func getMaterial(pixel uint32) string {
	if pixel == 0 {
		return "Empty"
	}
	// alpha is unused
	r, g, b, _ := toRGBA(uint32(pixel))

	rgba := []uint8{r, g, b}
	max := findMax(rgba)

	// TODO: is empty 0 or 255? the alpha channel might be 255 so double check

	if pixel == 0 {
		return "Empty"
	} else if r == max {
		return "Food"
	} else if g == max {
		return "Wall"
	} else if b == max {
		return "Water"
	} else {
		fmt.Println("can't match material for pixel", r, g, b)
		return "Wall"
	}
}

func findMax(xs []uint8) (m uint8) {
	for _, x := range xs {
		if x > m {
			m = x
		}
	}
	return
}

func toRGBA(color uint32) (red, green, blue, alpha uint8) {
	alpha = uint8(color & 0xFF)
	blue = uint8((color >> 8) & 0xFF)
	green = uint8((color >> 16) & 0xFF)
	red = uint8((color >> 24) & 0xFF)
	return
}

// rayTrace finds the first non empty pixel going in angle direction
func (w *World) rayTrace(angle int) (distance float32, pattern *etensor.Float32) {
	v := angVec(angle)
	// TODO: double check all these x and ys to make sure they correspond to row, col
	startX := float32(w.AgentX)
	startY := float32(w.AgentY)

	for {
		startX += v.X
		startY += v.Y
		distance++
		pixel, err := w.getPoint(int(startX), int(startY))
		if err != nil {
			fmt.Println("reached out of bounds ", startX, startY)
			return
		}
		material := getMaterial(pixel)
		if material == "Empty" {
			continue
		}
		pattern = w.Pats[material]
		return
	}
}

// for each angle between min and max, do ray trace
func (w *World) fillFOV(minAngle, maxAngle, step int) (distances []float32, patterns []*etensor.Float32) {
	maxld := mat32.Log(1 + mat32.Sqrt(float32(w.WorldX*w.WorldY+w.WorldY*w.WorldY)))

	for a := minAngle; a <= maxAngle; a += step {
		d, p := w.rayTrace(a)
		d = logMax(d, maxld)
		distances = append(distances, d)
		patterns = append(patterns, p)
	}

	return
}

func (w *World) PopCode(depthLogs []float32, dv *etensor.Float32, dvr *etensor.Float32, n int) etensor.Tensor {
	np := w.DepthSize / w.DepthPools

	for i := 0; i < n; i++ {
		sv := dvr.SubSpace([]int{0, i}).(*etensor.Float32)
		w.DepthCode.Encode(&sv.Values, depthLogs[i], w.DepthSize, popcode.Set)
		for dp := 0; dp < w.DepthPools; dp++ {
			for pi := 0; pi < np; pi++ {
				ri := dp*np + pi
				dv.Set([]int{dp, i, pi, 0}, sv.Values[ri])
			}
		}
	}
	return dv
}

func (w *World) fillFovPatterns(fovPatterns []*etensor.Float32, fsz int) {
	for i := 0; i < fsz; i++ {
		sv := w.Fv.SubSpace([]int{0, i}).(*etensor.Float32)
		sv.CopyFrom(fovPatterns[i])
	}
}

// getProxSoma basically looks around all 4 directions
func (w *World) getProxSoma() *etensor.Float32 {
	ps := &etensor.Float32{}
	ps.SetShape([]int{1, 4, 2, 1}, nil, []string{"1", "Pos", "OnOff", "1"})
	ps.SetZeros()

	angs := []int{0, -90, 90, 180}

	for i := 0; i < 4; i++ {
		v := angVec(w.Angle + angs[i])
		_, gp := NextVecPoint(w.PosF, v)
		pixel, _ := w.getPoint(gp.X, gp.Y)
		material := getMaterial(pixel)
		if material != "Empty" {
			ps.Set([]int{0, i, 0, 0}, 1) // on
		} else {
			ps.Set([]int{0, i, 1, 0}, 1) // off
		}
	}

	return ps
}

func (w *World) GetAllObservations() map[string]etensor.Tensor {
	obs := map[string]etensor.Tensor{}

	//states := []string{"Depth", "FovDepth", "Fovea", "ProxSoma", "Vestibular", "Inters", "Action", "Action"}
	//layers := []string{"V2Wd", "V2Fd", "V1F", "S1S", "S1V", "Ins", "VL", "Act"}

	wideDistances, _ := w.fillFOV(0, 180, 15)
	obs["V2Wd"] = w.PopCode(wideDistances, w.Dv, w.Dvr, w.NFOVRays)

	fsz := 1 + 2*w.FoveaSize
	fovDistances, fovPatterns := w.fillFOV(75, 105, 15)
	obs["V2Fd"] = w.PopCode(fovDistances, w.Fd, w.Fdr, fsz)

	w.fillFovPatterns(fovPatterns, fsz)
	obs["V1F"] = w.Fv

	obs["S1S"] = w.getProxSoma()

	// TODO: vestibular

	// TODO: inters

	// TODO: what is action and why is it twice?

	return obs
}

func angVec(ang int) mat32.Vec2 {
	a := mat32.DegToRad(float32(AngMod(ang)))
	v := mat32.Vec2{mat32.Cos(a), mat32.Sin(a)}
	return NormVecLine(v)
}

// NextVecPoint returns the next grid point along vector,
// from given current floating and grid points.  v is normalized
// such that the largest value is 1.
func NextVecPoint(cp, v mat32.Vec2) (mat32.Vec2, evec.Vec2i) {
	n := cp.Add(v)
	g := evec.NewVec2iFmVec2Round(n)
	return n, g
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

// AngMod returns angle modulo within 360 degrees
func AngMod(ang int) int {
	for ang < 0 {
		ang += 360
	}
	for ang > 360 {
		ang -= 360
	}
	return ang
}

func logMax(d, maxld float32) float32 {
	if d > 0 {
		return mat32.Log(1+d) / maxld
	} else {
		return 1
	}
}

// SaveWorld is a helper function that persists a world object to file
func SaveWorld(w World, filename gi.FileName) error {
	jenc, _ := json.MarshalIndent(w, "", " ")
	return ioutil.WriteFile(string(filename), jenc, 0644)
}

// OpenWorld loads a world object from file
func OpenWorld(filename gi.FileName) (w World) {
	fp, err := os.Open(string(filename))
	if err != nil {
		fmt.Println("ERROR! Can't find file", filename)
		return
	}
	defer fp.Close()
	b, err := ioutil.ReadAll(fp)
	err = json.Unmarshal(b, &w)
	if err != nil {
		fmt.Println(err)
	}
	return
}

// Config configures the bit pattern representations of mats and acts
func (w *World) Config() {
	w.AngInc = 15
	w.FOV = 180
	w.NFOVRays = (w.FOV / w.AngInc) + 1
	w.DepthSize = 32
	w.DepthPools = 8
	w.FoveaSize = 1

	w.Size.Set(w.WorldX, w.WorldY)
	w.PosI = w.Size.DivScalar(2) // start in middle -- could be random..
	w.PosF = w.PosI.ToVec2()

	w.DepthCode = popcode.OneD{}
	w.DepthCode.Defaults()
	w.DepthCode.SetRange(0.1, 1, 0.05)

	// TODO: buy some vowels

	w.Dv = &etensor.Float32{}
	w.Dv.SetShape([]int{w.DepthPools, w.NFOVRays, w.DepthSize / w.DepthPools, 1}, nil, []string{"Pools", "Angle", "Pop", "1"})

	w.Dvr = &etensor.Float32{}
	w.Dvr.SetShape([]int{1, w.NFOVRays, w.DepthSize, 1}, nil, []string{"1", "Angle", "Pop", "1"})

	fsz := 1 + 2*w.FoveaSize
	w.Fd = &etensor.Float32{}
	w.Fd.SetShape([]int{w.DepthPools, fsz, w.DepthSize / w.DepthPools, 1}, nil, []string{"Pools", "Angle", "Pop", "1"})

	w.Fdr = &etensor.Float32{}
	w.Fdr.SetShape([]int{1, fsz, w.DepthSize, 1}, nil, []string{"1", "Angle", "Pop", "1"})

	w.Fv = &etensor.Float32{}
	w.Fv.SetShape([]int{1, fsz, w.PatSize.Y, w.PatSize.X}, nil, []string{"1", "Angle", "Y", "X"})

	w.Pats = make(map[string]*etensor.Float32)
	w.PatSize.Set(5, 5)
	// TODO: list all the possible items egan can have
	w.Mats = []string{"Empty", "Wall", "Food", "Water", "FoodWas", "WaterWas"}

	for _, m := range w.Mats {
		t := &etensor.Float32{}
		t.SetShape([]int{w.PatSize.Y, w.PatSize.X}, nil, []string{"Y", "X"})
		w.Pats[m] = t
	}

	w.OpenPats("pats.json")
}

// OpenPats opens the patterns
func (w *World) OpenPats(filename gi.FileName) error {
	fp, err := os.Open(string(filename))
	if err != nil {
		fmt.Println("ERROR! Can't find pats file! This will prevent the model from working! :", err)
		return err
	}
	defer fp.Close()
	b, err := ioutil.ReadAll(fp)
	err = json.Unmarshal(b, &w.Pats)
	if err != nil {
		fmt.Println(err)
	}
	return err
}

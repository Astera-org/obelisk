package main

import (
	net_env "github.com/Astera-org/worlds/network/gengo/env"
	"github.com/emer/etable/etensor"
	"github.com/goki/mat32"
)

// TODO: add agent hunger and stuff (inters)

// this is actually the world from the agent's perspective
type World struct {
	Pixels         etensor.Tensor
	AgentX, AgentY int
}

func (w World) getPattern(point int) (pattern []float32) {

}

func (w World) getPoint(x, y int) (pattern []float32) {
	// TODO: check for out of bounds
	return
}

func (w World) rayTrace(angle int) (distance float32, pattern []float32) {
	v := angVec(angle)

	startX := float32(w.AgentX)
	startY := float32(w.AgentY)

	for {
		startX += v.X
		startY += v.Y
		distance++
		pattern = w.getPoint(int(startX), int(startY))
		if pattern != nil {
			return
		}
	}
}

// for each angle between min and max, do ray trace
func (w World) fillFOV(minAngle, maxAngle, step int) (distances []float32, patterns [][]float32) {
	for a := minAngle; a <= maxAngle; a += step {
		d, p := w.rayTrace(a)
		distances = append(distances, d)
		patterns = append(patterns, p)
	}
}

func (w World) GetAllObservations() map[string]*net_env.ETensor {
	obs := map[string]*net_env.ETensor{}

	wideDistances, _ := w.fillFOV(0, 180, 15)
	// TODO: popcode it

	fovDistances, fovPatterns := w.fillFOV(75, 105, 15)
	// TODO: popcode it

	// TODO: prox some

	// TODO: vestibular

	// TODO: inters

	// TODO: what is action and why is it twice?

	// These two lists line up which network layer corresponds to which state within FWorld.
	//states := []string{"Depth", "FovDepth", "Fovea", "ProxSoma", "Vestibular", "Inters", "Action", "Action"}
	//layers := []string{"V2Wd", "V2Fd", "V1F", "S1S", "S1V", "Ins", "VL", "Act"}
	//for i, lnm := range layers {
	//		obs[lnm] = network_agent.FromTensor(ev.State(states[i]))
	//}

	return obs
}

func angVec(ang int) mat32.Vec2 {
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

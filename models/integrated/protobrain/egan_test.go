package main

import (
	"fmt"
	"testing"
)

func TestRayTrace(t *testing.T) {

	w := OpenWorld("test_world.bin")
	w.Config()
	fmt.Println("Loaded world:", w)

	distance, pattern := w.rayTrace(0)
	fmt.Println("distance:", distance, "pattern:", pattern)
	t.Fatal()

	// TODO: test every angle in 15 increments

}

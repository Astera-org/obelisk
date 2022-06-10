package main

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"testing"
)

// TODO: add tests for each method

func TestRayTrace(t *testing.T) {

	w := OpenWorld("test_world.bin")
	w.Config()

	distance, pattern := w.rayTrace(0)
	fmt.Println("distance:", distance, "pattern:", pattern)

	expectedDistance := float32(1)
	assert.Equal(t, expectedDistance, distance, "Distance should be 1")

	// TODO: test every angle in 15 increments
}

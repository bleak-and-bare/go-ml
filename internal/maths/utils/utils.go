package utils

import (
	"math"

	"golang.org/x/exp/constraints"
)

func Sigmoid(z float64) float64 {
	return 1.0 / (1 + math.Exp(-z))
}

func Linspace(start float64, end float64, n int) []float64 {
	if end <= start || n <= 1 {
		return nil
	}

	var scale []float64
	step := float64(end-start) / float64(n-1)

	for k := range n {
		scale = append(scale, start+float64(k)*step)
	}

	return scale
}

// logspace with base=10
func Logspace(start float64, end float64, n int) []float64 {
	return LogspaceBase(start, end, n, 10)
}

func LogspaceBase(start float64, end float64, n int, base int) []float64 {
	if end <= start || n <= 1 {
		return nil
	}

	var scale []float64
	step := float64(end-start) / float64(n-1)

	for k := range n {
		scale = append(scale, math.Pow(float64(base), start+float64(k)*step))
	}
	return scale
}

// Returns 0 if null or return v/|v|
func Sign[T constraints.Float | constraints.Integer](v T) int {
	if v == 0 {
		return 0
	}

	if v < 0 {
		return -1
	}

	return 1
}

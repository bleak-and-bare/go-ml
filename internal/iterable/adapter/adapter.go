package adapter

import (
	"iter"
	"math"

	"github.com/bleak-and-bare/machine_learning/internal/iterable"
	"golang.org/x/exp/constraints"
)

// returns an iterator to each pointed value. Safely skip if encoutering nil.
func PtrDeref[T any](it iter.Seq[*T]) iter.Seq[T] {
	return func(yield func(T) bool) {
		for p := range it {
			if p != nil {
				if !yield(*p) {
					return
				}
			}
		}
	}
}

type Number interface {
	constraints.Float | constraints.Integer
}

// Returns absolute value of values
func Absolute[T Number](it iter.Seq[T]) iter.Seq[T] {
	return iterable.Map(it, func(v T) T {
		return T(math.Abs(float64(v)))
	})
}

// Returns an iterator of the squared values
func Squared[T Number](it iter.Seq[T]) iter.Seq[T] {
	return iterable.Map(it, func(v T) T {
		return v * v
	})
}

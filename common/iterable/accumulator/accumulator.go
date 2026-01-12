package accumulator

import (
	"iter"

	"golang.org/x/exp/constraints"
)

type Number interface {
	constraints.Float | constraints.Integer
}

// Accumulate iterable components
func Accumulate[T Number](v iter.Seq[T], a func(T, T) T) T {
	var acc T
	first := true

	for v := range v {
		if first {
			first = false
			acc = v
		} else {
			acc = a(acc, v)
		}
	}

	return acc
}

// Returns sum of iterable components
func Sum[T Number](v iter.Seq[T]) T {
	return Accumulate(v, func(a, b T) T {
		return a + b
	})
}

// Returns product of iterable components
func Product[T Number](v iter.Seq[T]) T {
	return Accumulate(v, func(a, b T) T {
		return a * b
	})
}

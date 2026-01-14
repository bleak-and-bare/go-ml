package vector

import (
	"iter"

	"github.com/bleak-and-bare/machine_learning/internal/iterable/accumulator"
	"golang.org/x/exp/constraints"
)

type Number interface {
	constraints.Float | constraints.Integer
}

// Returns an iterator for the transformed value of two iterators.
// Panics if the two iterables do not have the same length
func Transform[T Number](v, w iter.Seq[T], t func(T, T) T) iter.Seq[T] {
	return func(yield func(T) bool) {
		v_next, v_stop := iter.Pull(v)
		w_next, w_stop := iter.Pull(w)

		defer v_stop()
		defer w_stop()

		for {
			v, v_ok := v_next()
			w, w_ok := w_next()

			if v_ok != w_ok {
				panic("iterable.Transform : Vectors do not have the same length")
			}

			if !v_ok {
				break
			}

			if !yield(t(v, w)) {
				break
			}
		}
	}
}

func DotProduct[T Number](v, w iter.Seq[T]) T {
	return accumulator.Sum(Transform(v, w, func(v, w T) T {
		return v * w
	}))
}

// Returns an iterator for the difference of the two vectors
func Diff[T Number](v, w iter.Seq[T]) iter.Seq[T] {
	return Transform(v, w, func(v, w T) T {
		return v - w
	})
}

// Returns an iterator for the sum of the two vectors
func Sum[T Number](v, w iter.Seq[T]) iter.Seq[T] {
	return Transform(v, w, func(v, w T) T {
		return v + w
	})
}

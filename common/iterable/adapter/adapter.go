package adapter

import (
	"iter"

	"github.com/bleak-and-bare/machine_learning/common/iterable"
	"golang.org/x/exp/constraints"
)

func PtrDerefAdapter[T any](it iter.Seq[*T]) iter.Seq[T] {
	return func(yield func(T) bool) {
		for p := range it {
			if it != nil {
				if !yield(*p) {
					return
				}
			}
		}
	}
}

// Returns an iterator of the squared values
func Squared[T constraints.Float | constraints.Integer](it iter.Seq[T]) iter.Seq[T] {
	return iterable.Map(it, func(v T) T {
		return v * v
	})
}

package maths

import (
	"slices"

	"github.com/bleak-and-bare/machine_learning/internal/iterable"
	"github.com/bleak-and-bare/machine_learning/internal/iterable/accumulator"
	"github.com/bleak-and-bare/machine_learning/internal/iterable/adapter"
)

func Ridge[T Number](theta []T) T {
	return accumulator.Sum(adapter.Squared(iterable.Skip(slices.Values(theta), 1)))
}

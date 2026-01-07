package maths

import (
	"errors"
	"math"

	"golang.org/x/exp/constraints"
)

func L2Dist[T constraints.Float](v, w []T) (T, error) {
	if len(v) != len(w) {
		return 0.0, errors.New("MinkowskiDist : vectors do not have the same dimension")
	}

	var sum T
	for i := range v {
		diff := v[i] - w[i]
		sum += diff * diff
	}

	return T(math.Sqrt(float64(sum))), nil
}

package maths

import (
	"errors"
	"math"
)

type float interface {
	~float32 | ~float64
}

func L2Dist[T float](v, w []T) (T, error) {
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

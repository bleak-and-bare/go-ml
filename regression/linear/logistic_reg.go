package linear

import (
	"fmt"
	"math"

	"github.com/bleak-and-bare/machine_learning/internal/dataset"
	"github.com/bleak-and-bare/machine_learning/internal/iterable"
	"github.com/bleak-and-bare/machine_learning/internal/iterable/accumulator"
	"github.com/bleak-and-bare/machine_learning/internal/maths"
	"github.com/bleak-and-bare/machine_learning/internal/maths/regularization"
	"github.com/bleak-and-bare/machine_learning/internal/maths/utils"
	"golang.org/x/exp/constraints"
)

type LogisticRegression[T constraints.Float] struct {
	theta        []T     // parameter list
	Alpha        float32 // learning rate
	Penalty      regularization.RegularizationType
	Threshold    maths.Threshold
	hyper_params *struct { // regularization parameters
		alpha  float64 // L1-ratio
		lambda float64 // regularization strength
	}
}

func NewLogisticReg[T constraints.Float]() LogisticRegression[T] {
	return LogisticRegression[T]{
		Alpha:     1e-4,
		Threshold: maths.DefThreshold(),
		Penalty:   regularization.Ridge,
	}
}

func logreg_hypothesis[T constraints.Float](theta []T, sample dataset.DataSample[T]) (T, error) {
	d, err := sample.DotProduct(theta[1:])
	if err != nil {
		return 0.0, err
	}
	return T(utils.Sigmoid(float64(theta[0] + d))), nil
}

func scaled_negative_log_likelihood[T constraints.Float](theta []T, ds *dataset.DataSet[T]) (T, error) {
	var caught_err error

	s := -accumulator.Mean(iterable.Map(ds.Samples(), func(sample dataset.DataSample[T]) T {
		if caught_err != nil {
			return 0.0
		}

		y := sample.GetTarget()
		if y == nil {
			caught_err = fmt.Errorf("scaled_negative_log_likelihood : No target found for row : %v", sample.GetRow())
			return 0.0
		}

		h, err := logreg_hypothesis(theta, sample)
		if err != nil {
			caught_err = err
			return 0.0
		}

		return *y*T(math.Log(float64(h))) + (1-*y)*T(math.Log(float64(1-h)))
	}))

	if caught_err != nil {
		return 0.0, caught_err
	}

	return s, nil
}

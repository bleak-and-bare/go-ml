package linear

import (
	"fmt"
	"math"
	"slices"

	"github.com/bleak-and-bare/go-ml/machine_learning/common/dataset"
	"github.com/bleak-and-bare/go-ml/machine_learning/common/iterable"
	"github.com/bleak-and-bare/go-ml/machine_learning/common/iterable/accumulator"
	"github.com/bleak-and-bare/go-ml/machine_learning/common/maths"
	"github.com/bleak-and-bare/go-ml/machine_learning/common/maths/regularization"
	"github.com/bleak-and-bare/go-ml/machine_learning/common/maths/utils"
	"github.com/bleak-and-bare/go-ml/machine_learning/common/maths/vector"
	"golang.org/x/exp/constraints"
)

type LogisticRegression[T constraints.Float] struct {
	BaseModel[T]
	scaffolded  bool
	Alpha       float32 // learning rate
	Penalty     regularization.RegularizationType
	Threshold   maths.Threshold
	HyperParams *regularization.ElasticnetParams
}

func NewLogisticReg[T constraints.Float]() LogisticRegression[T] {
	return LogisticRegression[T]{
		Alpha:     1e-4,
		Threshold: maths.DefThreshold(),
		Penalty:   regularization.None,
	}
}

func (m *LogisticRegression[T]) SetHyperParams(alpha, lambda float64) {
	if m.HyperParams == nil {
		m.HyperParams = &regularization.ElasticnetParams{}
	}

	m.HyperParams.Alpha = alpha
	m.HyperParams.Lambda = lambda
}

func (m *LogisticRegression[T]) Predict(x []T) (T, error) {
	return m.BaseModel.Predict(x, logreg_hypothesis)
}

func (m *LogisticRegression[T]) PredictOn(ds *dataset.DataSet[T]) []T {
	return m.BaseModel.PredictOn(ds, logreg_hypothesis)
}

func logreg_hypothesis[T constraints.Float](theta []T, x []T) T {
	return T(utils.Sigmoid(float64(vector.DotProduct(slices.Values(theta), iterable.Prepend(slices.Values(x), 1.0)))))
}

func logreg_hypothesis_sample[T constraints.Float](theta []T, sample dataset.DataSample[T]) (T, error) {
	d, err := sample.DotProduct(theta[1:])
	if err != nil {
		return 0.0, err
	}

	h := T(utils.Sigmoid(float64(theta[0] + d)))
	return h, nil
}

func scaled_negative_log_likelihood[T constraints.Float](theta []T, ds *dataset.DataSet[T]) (T, error) {
	var caught_err error

	s := -accumulator.Mean(iterable.Map(ds.Samples(), func(sample dataset.DataSample[T]) T {
		if caught_err != nil {
			return 0.0
		}

		y := sample.GetTarget()
		if y == nil {
			caught_err = fmt.Errorf("scaled_negative_log_likelihood : No target found for row %v", sample.GetRow())
			return 0.0
		}

		d, err := sample.DotProduct(theta[1:])
		if err != nil {
			caught_err = err
			return 0.0
		}

		z := theta[0] + d
		return T(math.Log(1+math.Exp(float64(z)))) - *y*z
	}))

	if caught_err != nil {
		return 0.0, caught_err
	}

	if math.IsNaN(float64(s)) || math.IsInf(float64(s), 0) {
		return 0.0, fmt.Errorf("scaled_negative_log_likelihood : caught NaN or Inf value. Parameters : %v\n", theta)
	}

	return s, nil
}

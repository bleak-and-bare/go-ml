package maths

import (
	"fmt"

	"github.com/bleak-and-bare/go-ml/machine_learning/common/dataset"
	"github.com/bleak-and-bare/go-ml/machine_learning/common/iterable"
	"github.com/bleak-and-bare/go-ml/machine_learning/common/iterable/accumulator"
	"github.com/bleak-and-bare/go-ml/machine_learning/common/iterable/adapter"
	"golang.org/x/exp/constraints"
)

type MSE[T constraints.Float] struct {
	Hypothesis SampleFunction[T]
}

/*
Mean squared error
Parameters :
- h : hypothesis function
- placeholder : default value for invalid cells found in the dataset (does not matter if dataset have no holes)
*/
func (mse *MSE[T]) On(params []T, ds *dataset.DataSet[T]) (T, error) {
	var caught_err error
	m := accumulator.Mean(adapter.Squared(iterable.Map(ds.Samples(), func(ds dataset.DataSample[T]) T {
		if caught_err != nil {
			return 0.0
		}

		h, err := mse.Hypothesis.On(params, &ds)
		if err != nil {
			caught_err = err
			return 0.0
		}
		return h - *ds.GetTarget()
	})))

	if caught_err != nil {
		return 0.0, caught_err
	}

	return m, nil
}

func (mse *MSE[T]) Diff(j int, params []T, ds *dataset.DataSet[T]) (T, error) {
	sample_size := ds.Size()

	var sum T
	var caught_err error

	for ds := range ds.Samples() {
		y := ds.GetTarget()

		if y != nil {
			hypo, err := mse.Hypothesis.On(params, &ds)
			if err != nil {
				caught_err = err
				break
			}

			diff, err := mse.Hypothesis.Diff(j, params, &ds)
			if err != nil {
				caught_err = err
				break
			}

			sum += diff * (hypo - *y)
		} else {
			caught_err = fmt.Errorf("Target not found at <%d, %d>", ds.GetRow(), j-1)
			break
		}
	}

	if caught_err != nil {
		return 0.0, caught_err
	}

	return T(2/float32(sample_size)) * sum, nil
}

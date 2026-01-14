package optimization

import (
	"fmt"

	"github.com/bleak-and-bare/machine_learning/internal/dataset"
	"github.com/bleak-and-bare/machine_learning/internal/iterable"
	"github.com/bleak-and-bare/machine_learning/internal/iterable/accumulator"
	"github.com/bleak-and-bare/machine_learning/internal/iterable/adapter"
	"golang.org/x/exp/constraints"
)

type Hypothesis[T constraints.Float] interface {
	// Compute hypothesis value on given data sample
	// parameters :
	// - params : commonly called theta. With n+1 length where theta0 is the bias
	// - sample : one sample used for estimation
	On(params []T, sample *dataset.DataSample[T]) (T, error)

	// Compute hypothesis value on given data sample
	// parameters :
	// - j : index of theta variable to apply derivative. j == 0 means derivating regarding the bias theta0 and pulling the first feature with sample.GetFeat(0) might not what you want in that regard
	// - params : commonly called theta
	// - sample : one sample used for estimation
	Diff(j int, params []T, sample *dataset.DataSample[T]) (T, error)
}

func PartialDiffMSE[T constraints.Float](j int, params []T, ds *dataset.DataSet[T], h Hypothesis[T]) (T, error) {
	sample_size := ds.Size()

	var sum T
	var caught_err error

	for ds := range ds.Samples() {
		y := ds.GetTarget()

		if y != nil {
			hypo, err := h.On(params, &ds)
			if err != nil {
				caught_err = err
				break
			}

			diff, err := h.Diff(j, params, &ds)
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

/*
Mean squared error
Parameters :
- h : hypothesis function
- placeholder : default value for invalid cells found in the dataset
*/
func MSE[T constraints.Float](params []T, ds *dataset.DataSet[T], h func(params []T, sample []T) T, placeholder T) T {
	return accumulator.Mean(adapter.Squared(iterable.Map(ds.Samples(), func(ds dataset.DataSample[T]) T {
		return h(params, ds.GetSampleTestNoErr(placeholder)) - *ds.GetTarget()
	})))
}

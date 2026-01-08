package optimization

import (
	"fmt"

	"github.com/bleak-and-bare/machine_learning/common/dataset"
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

	if !ds.ForEachSample(func(ds dataset.DataSample[T]) bool {
		y := ds.GetTarget()

		if y != nil {
			hypo, err := h.On(params, &ds)
			if err != nil {
				caught_err = err
				return false
			}

			diff, err := h.Diff(j, params, &ds)
			if err != nil {
				caught_err = err
				return false
			}

			sum += diff * (hypo - *y)
			return true
		}

		caught_err = fmt.Errorf("Target not found at <%d, %d>", ds.GetRow(), j-1)
		return false
	}) {
		return 0.0, caught_err
	}

	return T(2/float32(sample_size)) * sum, nil
}

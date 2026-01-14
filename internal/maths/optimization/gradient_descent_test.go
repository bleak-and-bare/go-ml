package optimization

import (
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/bleak-and-bare/machine_learning/internal/dataset"
	"github.com/bleak-and-bare/machine_learning/internal/iterable"
	"github.com/bleak-and-bare/machine_learning/internal/maths"
	"github.com/bleak-and-bare/machine_learning/internal/maths/vector"
	"golang.org/x/exp/constraints"
)

// Copied from linear_reg.go

type linear_reg_hypo[T constraints.Float] struct{}

func (h *linear_reg_hypo[T]) On(params []T, sample *dataset.DataSample[T]) (T, error) {
	d, err := sample.DotProduct(params[1:])
	if err != nil {
		return 0.0, err
	}

	return params[0] + d, nil
}

func (h *linear_reg_hypo[T]) Diff(j int, params []T, sample *dataset.DataSample[T]) (T, error) {
	if j == 0 {
		return 1, nil
	}

	x := sample.GetFeat(j - 1)
	if x == nil {
		return 0.0, fmt.Errorf("No feature found at <%d, %d>", sample.GetRow(), j-1)
	}

	return *x, nil
}

func linear_reg_cost_partial_diff[T constraints.Float](j int, theta []T, ds *dataset.DataSet[T]) (T, error) {
	var h linear_reg_hypo[T]
	return PartialDiffMSE(j, theta, ds, &h)
}

func linear_reg_cost[T constraints.Float](theta []T, ds *dataset.DataSet[T]) T {
	return MSE(theta, ds, func(theta []T, x []T) T {
		return theta[0] + vector.DotProduct(
			iterable.Skip(slices.Values(theta), 1),
			slices.Values(x),
		)
	}, 0.0)
}

func TestGradientDescent(t *testing.T) {
	ds := dataset.NewDataSet[float32](1)
	str := strings.NewReader(`x,y
0,0
1,1
2,2`)

	ds.LoadCsvReader(str, ',')
	sgd := NewSGD[float32](maths.DefThreshold())
	sgd.Cost = linear_reg_cost
	sgd.CostPartialDiff = linear_reg_cost_partial_diff

	if err := sgd.Fit(&ds); err != nil {
		t.Errorf("SGD.Fit should not error : %v", err)
		t.FailNow()
	}

	theta := sgd.GetParams()
	if len(theta) != 2 {
		t.Error("Wrong number of parameters")
	}

	if math.Abs(float64(theta[0])) >= 1e-3 || math.Abs(float64(theta[1]-1)) >= 1e-3 {
		t.Errorf("Wrong parameter values : [%.3f, %.3f] != [1, 0]", theta[1], theta[0])
	}
}

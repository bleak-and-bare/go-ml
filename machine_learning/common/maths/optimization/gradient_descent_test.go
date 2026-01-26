package optimization

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/bleak-and-bare/go-ml/machine_learning/common/dataset"
	"github.com/bleak-and-bare/go-ml/machine_learning/common/maths"
	"golang.org/x/exp/constraints"
)

// Copied from linear_reg_fit.go

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

func TestGradientDescent(t *testing.T) {
	ds := dataset.NewDataSet[float32](1)
	str := strings.NewReader(`x,y
0,0
1,1
2,2`)

	ds.LoadCsvReader(str, ',')
	sgd := NewSGD[float32](maths.DefThreshold())
	sgd.Cost = &maths.MSE[float32]{
		Hypothesis: &linear_reg_hypo[float32]{},
	}

	if err := sgd.Fit(&ds); err != nil {
		t.Errorf("SGD.Fit should not error : %v", err)
		t.FailNow()
	}

	theta := sgd.GetParams()
	if len(theta) != 2 {
		t.Error("Wrong number of parameters")
	}

	if math.Abs(float64(theta[0])) >= 1e-1 || math.Abs(float64(theta[1]-1)) >= 1e-1 {
		t.Errorf("Wrong parameter values : [%.3f, %.3f] != [1, 0]", theta[1], theta[0])
	}
}

package linear

import (
	"math"
	"strings"
	"testing"

	"github.com/bleak-and-bare/machine_learning/common/dataset"
)

func TestLinearRegression_Fit(t *testing.T) {
	ds := dataset.NewDataSet[float32](1)
	str := strings.NewReader(`x,y
0,0
1,1
2,2`)

	ds.LoadCsvReader(str, ',')
	m := NewLinearReg[float32]()

	if err := m.Fit(&ds); err != nil {
		t.Error("LinearRegression.Fit should not error")
		t.FailNow()
	}

	for i := range 3 {
		pred, err := m.Predict([]float32{float32(i)})
		if err != nil {
			t.Errorf("LinearRegression.Predict(%d) should not error: %v", i, err)
			t.FailNow()
		}

		if int(math.Round(float64(pred))) != i {
			t.Errorf("LinearRegression.Predict(%d) != %.3f", i, pred)
		}
	}
}

func TestLinearRegression_PredictOn(t *testing.T) {
	ds := dataset.NewDataSet[float32](1)
	str := strings.NewReader(`x,y
0,0
1,1
2,2`)

	ds.LoadCsvReader(str, ',')
	m := NewLinearReg[float32]()

	if err := m.Fit(&ds); err != nil {
		t.Error("LinearRegression.Fit should not error")
		t.FailNow()
	}

	report := m.PredictOn(&ds)
	if report.SkippedRows > 0 {
		t.Error("No rows should have been skipped")
	}

	if report.Score < 0.9 {
		t.Error("Bad score")
	}
}

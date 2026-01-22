package main

import (
	"fmt"
	"math"
	"os"

	"github.com/bleak-and-bare/machine_learning/classification"
	"github.com/bleak-and-bare/machine_learning/internal/dataset"
	"github.com/bleak-and-bare/machine_learning/internal/maths/regularization"
	"github.com/bleak-and-bare/machine_learning/processing"
	"github.com/bleak-and-bare/machine_learning/regression/linear"
)

func main() {
	ds := dataset.NewDataSet[float32](7)
	if err := ds.LoadCsv("./examples/dataset/Raisin_Dataset.csv", ','); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load dataset : %v", err)
		return
	}

	ds.MapColumn("Class", func(dc dataset.DataCell) dataset.DataCell {
		col, _ := dc.(*dataset.StrDataCell)
		if col.Value == "Kecimen" {
			return &dataset.RealDataCell[float32]{Value: 1.0}
		}

		return &dataset.RealDataCell[float32]{Value: 0.0}
	})

	ds.Shuffle()
	train, _ := ds.Extract(0.0, 0.75)
	test, _ := ds.Extract(0.75, 1.0)

	scaler := processing.StandardScaler[float32]{}
	cols := test.GetColumnNames()
	for _, col := range cols {
		if col != "Class" {
			scaler.FitTransformDataSet(train, col)
			scaler.TransformDataSet(test, col)
		}
	}

	m := linear.NewLogisticReg[float32]()
	m.Penalty = regularization.None
	if err := m.Fit(train); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to train model : %v", err)
		return
	}

	// fmt.Printf("m.GetParams(): %v\n", m.GetParams())

	t := test.CollectTargets()
	r := m.PredictOn(test)

	trg := make([]int, len(t))
	pred := make([]int, len(r))

	for i := range r {
		trg[i] = int(t[i])
		pred[i] = int(math.Round(float64(r[i])))
	}

	classification.ComputeReport(trg, pred)
}

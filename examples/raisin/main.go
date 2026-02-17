package main

import (
	"fmt"
	"math"
	"os"
	"slices"

	"github.com/bleak-and-bare/go-ml/machine_learning/classification"
	"github.com/bleak-and-bare/go-ml/machine_learning/common/dataset"
	"github.com/bleak-and-bare/go-ml/machine_learning/common/iterable/accumulator"
	"github.com/bleak-and-bare/go-ml/machine_learning/common/maths/regularization"
	"github.com/bleak-and-bare/go-ml/machine_learning/common/maths/utils"
	"github.com/bleak-and-bare/go-ml/machine_learning/common/misc"
	"github.com/bleak-and-bare/go-ml/machine_learning/common/selector"
	"github.com/bleak-and-bare/go-ml/machine_learning/processing"
	"github.com/bleak-and-bare/go-ml/machine_learning/regression/linear"
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

	// chunk, _ := ds.Extract(0.0, 0.125)
	// ds = *chunk

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

	train.Head(5, "Raisin train dataset", true)

	m := linear.NewLogisticReg[float32]()
	m.Penalty = regularization.Ridge

	gs := selector.NewGridSearch(map[string][]float64{
		"lambda": utils.Logspace(-4, -2, 5),
	}, linear.LogRegFactory(m, func(m map[string]float64) regularization.ElasticnetParams {
		return regularization.ElasticnetParams{
			Lambda: m["lambda"],
		}
	}))

	if err := gs.Fit(train); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to search hyper params : %v", err)
		return
	}

	best_params := gs.BestParams()
	m.SetHyperParams(0.0, best_params["lambda"])

	if err := m.Fit(train); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to train model : %v", err)
		return
	}

	t := test.CollectTargets()
	r := m.PredictOn(test)

	var gp misc.GridPrinter
	gp.Columns("Min", "Max", "Mean")
	gp.NewRow()
	gp.Columns(
		fmt.Sprintf("%.3f", slices.Min(r)),
		fmt.Sprintf("%.3f", slices.Max(r)),
		fmt.Sprintf("%.3f", accumulator.Mean(slices.Values(r))),
	)
	gp.Print(true)

	trg := make([]int, len(t))
	pred := make([]int, len(r))

	for i := range r {
		trg[i] = int(t[i])
		pred[i] = int(math.Round(float64(r[i])))
	}

	classification.ComputeReport(trg, pred, true)
}

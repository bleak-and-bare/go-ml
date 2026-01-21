package main

import (
	"fmt"
	"os"
	"slices"

	"github.com/bleak-and-bare/machine_learning/internal/dataset"
	"github.com/bleak-and-bare/machine_learning/processing"
	"github.com/bleak-and-bare/machine_learning/regression/linear"
)

func main() {
	const pref_idx = 5
	ds := dataset.NewDataSet[float32](pref_idx)

	if err := ds.LoadCsv("./examples/dataset/Student_Performance.csv", ','); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load csv : %v", err)
		return
	}

	ds.MapColumn("Extracurricular Activities", func(c dataset.DataCell) dataset.DataCell {
		s, _ := c.(*dataset.StrDataCell)
		if s.Value == "Yes" {
			return &dataset.RealDataCell[float32]{Value: 1.0}
		}

		return &dataset.RealDataCell[float32]{Value: 0.0}
	})

	train, _ := ds.Extract(0.0, 0.75)
	test, _ := ds.Extract(0.75, 1.0)
	sample := []float32{8, 60, 0, 4, 7}

	var std_scaler processing.StandardScaler[float32]
	col_names := train.GetColumnNames()
	for i, col := range col_names {
		if col != "Extracurricular Activities" {
			std_scaler.FitTransformDataSet(train, col)
			std_scaler.TransformDataSet(test, col)

			if i < len(sample) {
				sample[i] = std_scaler.TransformOne(sample[i])
			}
		}
	}
	test.Head(5)

	m := linear.NewLinearReg[float32]()
	if err := m.Fit(train); err != nil {
		fmt.Fprintf(os.Stderr, "m.Fit errored : %v", err)
		return
	}

	pred, _ := m.Predict(sample)
	sample = append(sample, pred)
	sample = std_scaler.InverseTransform(slices.Values(sample))
	fmt.Printf("sample test : %v\n", sample[len(sample)-1])

	r := m.PredictOn(train)
	fmt.Printf("---------------------------------- Train set\n")
	fmt.Printf("Skipped rows = %v\n", r.SkippedRows)
	fmt.Printf("Mean absolute error = %v\n", r.MeanAbsoluteErr)
	fmt.Printf("RMSE = %v\n", r.RootMeanSquareErr)
	fmt.Printf("R2 Score = %v\n", r.Score)

	r = m.PredictOn(test)
	fmt.Printf("---------------------------------- Test set\n")
	fmt.Printf("Skipped rows = %v\n", r.SkippedRows)
	fmt.Printf("Mean absolute error = %v\n", r.MeanAbsoluteErr)
	fmt.Printf("RMSE = %v\n", r.RootMeanSquareErr)
	fmt.Printf("R2 Score = %v\n", r.Score)
	fmt.Printf("----------------------------------\n")
}

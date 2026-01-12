package main

import (
	"fmt"
	"os"

	"github.com/bleak-and-bare/machine_learning/internal/dataset"
	"github.com/bleak-and-bare/machine_learning/regression/linear"
)

func main() {
	ds := dataset.NewDataSet[float32](2)
	if err := ds.LoadCsv("../dataset/Salary_dataset.csv", ','); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load dataset : %v", err)
		return
	}

	ds.DropColumnAt(0).Head(5)
	train, _ := ds.Extract(0.0, 0.75)
	test, _ := ds.Extract(0.75, 1.0)

	m := linear.NewLinearReg[float32]()
	m.Alpha = 1e-2
	if err := m.Fit(train); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to fit model : %v", err)
		return
	}

	r := m.PredictOn(test)
	fmt.Printf("----------------------------------\n")
	fmt.Printf("Skipped rows = %v\n", r.SkippedRows)
	fmt.Printf("Mean absolute error = %v\n", r.MeanAbsoluteErr)
	fmt.Printf("RMSE = %v\n", r.RootMeanSquareErr)
	fmt.Printf("R2 Score = %v\n", r.Score)
}

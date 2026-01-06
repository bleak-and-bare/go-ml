package main

import (
	"fmt"
	"os"

	"github.com/bleak-and-bare/machine_learning/common/dataset"
	regression "github.com/bleak-and-bare/machine_learning/regression/linear"
)

func main() {
	ds := dataset.NewDataSet[float32](2)
	if err := ds.LoadCsv("./dataset/Salary_dataset.csv", ','); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load csv : %v", err)
		return
	}

	ds.DropColumnAt(0) // .Head(5)
	train, _ := ds.Extract(0.0, 0.8)
	test, _ := ds.Extract(0.8, 1.0)

	m := regression.NewLinearReg[float32]()
	m.Fit(train)
	m.PrintRegLineEquation()

	report := m.PredictOn(test)
	fmt.Printf("Skipped rows : %d\n", report.SkippedRows)
	fmt.Printf("Error : %.4f", report.Error)
}

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

	ds.DropColumnAt(0).Head(5)
	train, _ := ds.Extract(0.0, 0.8)
	test, _ := ds.Extract(0.8, 1.0)

	m := regression.NewLinearReg[float32]()
	m.Fit(train)
	m.PrintRegLineEq()

	var caught_err error
	fmt.Printf("predicted\texpected\n")

	if !test.ForEachSample(func(ds dataset.DataSample[float32]) bool {
		test := ds.GetSampleTest(0.0)
		y, err := m.Predict(test)
		if err != nil {
			caught_err = err
			return false
		}

		fmt.Printf("%.3f\t%.3f\n", y, *ds.GetTarget())

		return true
	}) {
		fmt.Fprintf(os.Stderr, "Failed to predict on test sample : %v", caught_err)
	}
}

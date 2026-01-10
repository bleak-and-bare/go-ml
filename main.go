package main

import (
	"fmt"
	"os"

	"github.com/bleak-and-bare/machine_learning/common/dataset"
)

func main() {
	ds := dataset.NewDataSet[float32](2)
	if err := ds.LoadCsv("./dataset/Student_Performance.csv", ','); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load csv : %v", err)
		return
	}

	ds.Head(10)
}

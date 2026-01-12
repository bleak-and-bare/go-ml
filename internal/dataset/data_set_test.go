package dataset_test

import (
	"slices"
	"strings"
	"testing"

	"github.com/bleak-and-bare/machine_learning/internal/dataset"
)

func mock_data_set() (dataset.DataSet[float32], error) {
	csv := `,YearsExperience,Salary
0,1.2000000000000002,39344.0
1,1.4000000000000001,46206.0
2,1.6,37732.0
3,2.1,43526.0
4,2.3000000000000003,39892.0
5,3.0,56643.0
6,3.1,60151.0
5,3.0,56643.0
1,1.4000000000000001,46206.0
3,2.1,43526.0
`
	ds := dataset.NewDataSet[float32](2)
	return ds, ds.LoadCsvReader(strings.NewReader(csv), ',')
}

func TestDataSet_LoadCsv(t *testing.T) {
	ds, err := mock_data_set()
	if err != nil {
		t.Errorf("Failed to load CSV : %v", err)
		return
	}

	cols := ds.GetColumnNames()
	if !slices.Equal(cols, []string{"", "YearsExperience", "Salary"}) {
		t.Error("Incorrect headers read")
	}

	count := ds.Size()
	if count != 10 {
		t.Error("Wrong number of row read")
	}
}

func TestDataSet_DropColumnAt(t *testing.T) {
	ds, _ := mock_data_set()
	cols := ds.DropColumnAt(0).GetColumnNames()

	if !slices.Equal(cols, []string{"YearsExperience", "Salary"}) {
		t.Error("Failed to drop column")
	}
}

func TestDataSet_DropColumn(t *testing.T) {
	ds, _ := mock_data_set()
	cols := ds.DropColumn("YearsExperience").GetColumnNames()

	if !slices.Equal(cols, []string{"", "Salary"}) {
		t.Error("Failed to drop column")
	}
}

func TestDataSet_Extract(t *testing.T) {
	ds, _ := mock_data_set()
	self, err := ds.Extract(0.0, 1.0)

	if err != nil {
		t.Error("Should be able to extract same range")
		t.FailNow()
	}

	if self.Size() != ds.Size() {
		t.Error("Extracting whole dataset produces different size")
	}

	test, err := ds.Extract(0.5, 1.0)
	if err != nil {
		t.Error("Should be able to extract with correct range")
		t.FailNow()
	}

	if ds.Size() != 2*test.Size() {
		t.Errorf("Invalid extracted size : %d != 2 * %d", ds.Size(), test.Size())
	}

	_, err = ds.Extract(10, 4)
	if err == nil {
		t.Errorf("Should not be able to extract : invalid range")
	}
}

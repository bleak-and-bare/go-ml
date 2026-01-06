package dataset

import (
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"slices"
	"strconv"
	"strings"
)

type header_t struct {
	name string
	used bool
}

type float interface {
	~float32 | ~float64
}

type DataSet[T float] struct {
	min_range         float32
	max_range         float32
	headers           []header_t
	real_feat_indices []int
	trg_col_idx       uint32
	datas             [][](*T)
}

func NewDataSet[T float](trg_col_idx uint32) DataSet[T] {
	var ds DataSet[T]
	ds.trg_col_idx = trg_col_idx
	ds.min_range = 0.0
	ds.max_range = 1.0
	return ds
}

func (ds *DataSet[T]) update_feat_indices() {
	var idx int
	for i := range ds.headers {
		if i != int(ds.trg_col_idx) && ds.headers[i].used {
			ds.real_feat_indices[idx] = i
			idx++
		}
	}

	for ; idx < len(ds.real_feat_indices); idx++ {
		ds.real_feat_indices[idx] = -1
	}
}

func (ds *DataSet[T]) FillEmpties(placeholder T) {
	for _, row := range ds.datas {
		for j := range row {
			if row[j] == nil {
				row[j] = &placeholder
			}
		}
	}
}

func (ds *DataSet[T]) FeatCount() int {
	return slices.Index(ds.real_feat_indices, -1)
}

func (ds *DataSet[T]) GetFeat(row int, feat int) *T {
	if feat >= len(ds.real_feat_indices) || feat < 0 || row < ds.min_bound() || row >= ds.max_bound() {
		return nil
	}

	idx := ds.real_feat_indices[feat]
	if idx == -1 {
		return nil
	}

	return ds.datas[row][idx]
}

func (ds *DataSet[T]) DropColumnAt(idx uint8) *DataSet[T] {
	if idx != uint8(ds.trg_col_idx) && int(idx) < len(ds.headers) {
		ds.headers[idx].used = false
		ds.update_feat_indices()
	}
	return ds
}

func (ds *DataSet[T]) DropColumn(name string) *DataSet[T] {
	i := slices.IndexFunc(ds.headers, func(h header_t) bool {
		return h.name == name
	})
	return ds.DropColumnAt(uint8(i))
}

func (ds *DataSet[T]) GetColumnNames() []string {
	cols := make([]string, 0, len(ds.headers))
	for _, h := range ds.headers {
		if h.used {
			cols = append(cols, h.name)
		}
	}
	return cols
}

func (ds *DataSet[T]) GetSample(i int) []*T {
	if i >= len(ds.datas) {
		return nil
	}

	return ds.datas[i]
}

func (ds *DataSet[T]) Extract(min_range float32, max_range float32) (*DataSet[T], error) {
	if min_range > max_range {
		return nil, errors.New("DataSet.Extract : invalid range provided")
	}

	new_ds := *ds
	ds_range := ds.max_range - ds.min_range
	new_ds.min_range = max(0.0, ds.min_range+ds_range*min_range)
	new_ds.max_range = min(1.0, ds.min_range+ds_range*max_range)

	return &new_ds, nil
}

// returns real samples count
func (ds *DataSet[T]) raw_count() uint32 {
	return uint32(len(ds.datas))
}

func (ds *DataSet[T]) min_bound() int {
	row_count := ds.raw_count()
	return int(math.Floor(float64(ds.min_range * float32(row_count))))
}

func (ds *DataSet[T]) max_bound() int {
	row_count := ds.raw_count()
	return min(int(row_count), int(math.Round(float64(ds.max_range*float32(row_count)))))
}

func (ds *DataSet[T]) Size() uint32 {
	rng := float64(ds.max_range - ds.min_range)
	return uint32(math.Round(rng * float64(ds.raw_count())))
}

func (ds *DataSet[T]) Empty() bool {
	return ds.Size() == 0
}

func (ds *DataSet[T]) Head(max uint8) {
	if ds.datas == nil {
		return
	}

	var line_sep strings.Builder
	for _, h := range ds.headers {
		if !h.used {
			continue
		}

		line_sep.WriteString(strings.Repeat("-", len(h.name)+8))
		fmt.Printf("%v\t\t", h.name)
	}
	fmt.Printf("\n%v\n", line_sep.String())

	max_bound := ds.max_bound()
	for i := ds.min_bound(); i < max_bound && max > 0; i, max = i+1, max-1 {
		row := ds.datas[i]

		for j, col := range row {
			if j >= len(ds.headers) || !ds.headers[j].used {
				continue
			}

			if col != nil {
				fmt.Printf("%.3f\t\t", *col)
			} else {
				fmt.Printf("\t\t\t\t")
			}
		}
		fmt.Println("")
	}
}

func (ds *DataSet[T]) Dump() {
	ds.Head(uint8(ds.raw_count()))
}

func (ds *DataSet[T]) LoadCsvReader(input_reader io.Reader, delim rune) error {
	reader := csv.NewReader(input_reader)
	reader.Comma = delim

	first_line := true
	for {
		cols, err := reader.Read()
		if err != nil {
			if err == io.EOF {
				break
			}
			return err
		}

		if first_line {
			first_line = false
			for _, col := range cols {
				ds.headers = append(ds.headers, header_t{col, true})
			}
		} else if len(cols) > 0 {
			ds.datas = append(ds.datas, make([]*T, 0, len(ds.headers)))
			row := ds.datas[len(ds.datas)-1]

			for i, col := range cols {
				if i >= len(ds.headers) { // make sure every row has the same length as header
					break
				}

				if c, err := strconv.ParseFloat(col, 64); err == nil {
					concrete := T(c)
					row = append(row, &concrete)
				} else {
					row = append(row, nil)
				}
			}

			r := len(ds.headers) - len(cols)
			if r > 0 {
				for ; r > 0; r-- {
					row = append(row, nil)
				}
			}

			ds.datas[len(ds.datas)-1] = row
		}
	}

	ds.real_feat_indices = make([]int, len(ds.headers)-1)
	ds.update_feat_indices()

	return nil
}

func (ds *DataSet[T]) LoadCsv(path string, delim rune) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	return ds.LoadCsvReader(file, delim)
}

func (ds *DataSet[T]) ForEachSample(cb func(DataSample[T]) bool) bool {
	max_bound := ds.max_bound()
	for i := ds.min_bound(); i < max_bound; i++ {
		if !cb(DataSample[T]{
			owner: ds,
			row:   i,
		}) {
			return false
		}
	}
	return true
}

func (ds *DataSet[T]) TargetVariance() float64 {
	n := ds.max_bound()
	sum := 0.0

	for i := range n {
		for j := i + 1; j < n; j++ {
			y_i := ds.datas[i][ds.trg_col_idx]
			y_j := ds.datas[j][ds.trg_col_idx]

			if y_i == nil || y_j == nil {
				continue
			}

			diff := *y_i - *y_j
			sum += float64(diff * diff)
		}
	}

	return sum / float64(n*n)
}

func (ds *DataSet[T]) Shuffle() *DataSet[T] {
	start := ds.min_bound()
	real_size := int(ds.raw_count())

	rand.Shuffle(int(ds.Size()), func(i, j int) {
		if start+i < real_size && start+j < real_size {
			ds.datas[start+i], ds.datas[start+j] = ds.datas[start+j], ds.datas[start+i]
		}
	})

	return ds
}

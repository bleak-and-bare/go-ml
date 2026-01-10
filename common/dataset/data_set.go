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

	"golang.org/x/exp/constraints"
)

type header_t struct {
	name string
	used bool
}

type DataSet[T constraints.Float] struct {
	min_range         float32
	max_range         float32
	headers           []header_t
	real_feat_indices []int
	trg_col_idx       uint32
	datas             []DataCell // row major
}

func NewDataSet[T constraints.Float](trg_col_idx uint32) DataSet[T] {
	var ds DataSet[T]
	ds.trg_col_idx = trg_col_idx
	ds.min_range = 0.0
	ds.max_range = 1.0
	return ds
}

func (ds *DataSet[T]) Copy() DataSet[T] {
	copy := *ds
	copy.headers = slices.Clone(ds.headers)
	copy.real_feat_indices = slices.Clone(ds.real_feat_indices)
	copy.datas = slices.Clone(ds.datas)
	return copy
}

func (ds *DataSet[T]) MapColumn(i int, cb func(DataCell) T) *DataSet[T] {
	if i < 0 || i >= len(ds.headers) {
		return ds
	}

	start, end := ds.min_bound(), ds.max_bound()
	for row := start; row < end; row++ {
		ds.set_at(row, i, cb(ds.at(row, i)))
	}

	return ds
}

func (ds *DataSet[T]) set_at(i, j int, v T) {
	ds.datas[i*len(ds.headers)+j] = &RealDataCell[T]{v}
}

func (ds *DataSet[T]) at(i, j int) DataCell {
	return ds.datas[i*len(ds.headers)+j]
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

func (ds *DataSet[T]) SetTargetCol(i int) error {
	if i == int(ds.trg_col_idx) {
		return nil
	}

	if i < 0 || i >= len(ds.headers) {
		return fmt.Errorf("Invalid column index : %d", i)
	}

	if !ds.headers[i].used {
		return errors.New("Can not set unused column as target")
	}

	return nil
}

func (ds *DataSet[T]) FillEmpties(placeholder T) {
	for i := range ds.datas {
		if ds.datas[i] == nil {
			ds.datas[i] = &RealDataCell[T]{placeholder}
		}
	}
}

func (ds *DataSet[T]) FeatCount() int {
	if end := slices.Index(ds.real_feat_indices, -1); end != -1 {
		return end
	}
	return len(ds.real_feat_indices)
}

func (ds *DataSet[T]) GetFeat(row int, feat int) *T {
	if feat >= len(ds.real_feat_indices) || feat < 0 || row < ds.min_bound() || row >= ds.max_bound() {
		return nil
	}

	idx := ds.real_feat_indices[feat]
	if idx == -1 {
		return nil
	}

	cell := ds.at(row, idx)
	if c, ok := cell.(*RealDataCell[T]); ok {
		return &c.Value
	}

	return nil
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
	return uint32(len(ds.datas) / len(ds.headers))
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

func (ds *DataSet[T]) Head(max uint32) {
	if ds.datas == nil || max == 0 {
		return
	}

	const tab = "  "
	max_bound := ds.max_bound()
	visited := make([]bool, len(ds.headers))
	max_lengths := make([]int, len(ds.headers))

	for i, h := range ds.headers {
		if !h.used {
			continue
		}

		if !visited[i] {
			visited[i] = true
			max_lengths[i] = len(h.name)
		} else if max_lengths[i] < len(h.name) {
			max_lengths[i] = len(h.name)
		}
	}

	for i := ds.min_bound(); i < max_bound; i++ {
		for j := range ds.headers {
			if !ds.headers[j].used {
				continue
			}

			col := ds.at(i, j)
			switch c := col.(type) {
			case *RealDataCell[T]:
				s := fmt.Sprintf("%.3f", c.Value)
				l := len(s)
				if l > max_lengths[j] {
					max_lengths[j] = l
				}
			case *StrDataCell:
				l := len(c.Value)
				if l > max_lengths[j] {
					max_lengths[j] = l
				}
			}
		}
	}

	var line_sep strings.Builder
	for i, h := range ds.headers {
		if !h.used {
			continue
		}

		line_sep.WriteString(strings.Repeat("-", max_lengths[i]+len(tab)+1))
		fmt.Printf("%v%v%v|", h.name, strings.Repeat(" ", max_lengths[i]-len(h.name)), tab)
	}
	fmt.Printf("\n%v\n", line_sep.String())

	for i := ds.min_bound(); i < max_bound && max > 0; i, max = i+1, max-1 {
		for j := range ds.headers {
			if j >= len(ds.headers) || !ds.headers[j].used {
				continue
			}

			col := ds.at(i, j)
			str := ""

			switch c := col.(type) {
			case *RealDataCell[T]:
				str = fmt.Sprintf("%.3f", c.Value)
			case *StrDataCell:
				str = c.Value
			}

			fmt.Printf("%v%v%v|", str, strings.Repeat(" ", max_lengths[j]-len(str)), tab)
		}
		fmt.Println("")
	}
}

func (ds *DataSet[T]) Dump() {
	ds.Head(ds.raw_count())
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
			ds.datas = slices.Grow(ds.datas, len(ds.headers))

			for i, col := range cols {
				if i >= len(ds.headers) { // make sure every row has the same length as header
					break
				}

				if c, err := strconv.ParseFloat(col, 64); err == nil {
					concrete := T(c)
					ds.datas = append(ds.datas, &RealDataCell[T]{concrete})
				} else {
					ds.datas = append(ds.datas, &StrDataCell{col})
				}
			}

			r := len(ds.headers) - len(cols)
			if r > 0 {
				for ; r > 0; r-- {
					ds.datas = append(ds.datas, nil)
				}
			}
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

// Compute variance of the target column.
// This method will skip any non-real cells
func (ds *DataSet[T]) TargetVariance() float64 {
	n := ds.max_bound()
	sum := 0.0

	for i := range n {
		for j := i + 1; j < n; j++ {
			y_i := ds.at(i, int(ds.trg_col_idx))
			y_j := ds.at(j, int(ds.trg_col_idx))

			if y_i == nil || y_j == nil || !y_i.IsReal() || !y_j.IsReal() {
				continue
			}

			yi, _ := y_i.(*RealDataCell[T])
			yj, _ := y_j.(*RealDataCell[T])

			diff := yi.Value - yj.Value
			sum += float64(diff * diff)
		}
	}

	return sum / float64(n*n)
}

func (ds *DataSet[T]) Shuffle() *DataSet[T] {
	start := ds.min_bound()
	cols := len(ds.headers)
	real_size := int(ds.raw_count())

	rand.Shuffle(int(ds.Size()), func(i, j int) {
		if start+i < real_size && start+j < real_size {
			for k := range cols {
				ds.datas[i*cols+k], ds.datas[j*cols+k] = ds.datas[j*cols+k], ds.datas[i*cols+k]
			}
		}
	})

	return ds
}

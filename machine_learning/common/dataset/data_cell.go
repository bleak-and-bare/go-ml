package dataset

import (
	"golang.org/x/exp/constraints"
)

type DataCell interface {
	IsReal() bool
}

type StrDataCell struct {
	Value string
}

func (c *StrDataCell) IsReal() bool {
	return false
}

type RealDataCell[T constraints.Float] struct {
	Value T
}

func (c *RealDataCell[T]) IsReal() bool {
	return true
}

package dataset

import (
	"fmt"

	"golang.org/x/exp/constraints"
)

type DataCell interface {
	fmt.Stringer
	IsReal() bool
}

type StrDataCell struct {
	Value string
}

func (c *StrDataCell) String() string {
	return c.Value
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

func (c *RealDataCell[T]) String() string {
	return fmt.Sprintf("%.3f", c.Value)
}

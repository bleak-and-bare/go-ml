package selector

import "golang.org/x/exp/constraints"

type Number interface {
	constraints.Float | constraints.Integer
}

// Choose hyper parameter based on estimator using cross validation
type GridSearch[T Number] struct {
	params map[string]T
}

package regularization

type RegularizationType int

const (
	ElasticNet RegularizationType = iota
	Lasso
	Ridge
	None
)

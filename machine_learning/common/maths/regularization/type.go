package regularization

type RegularizationType int

const (
	None RegularizationType = iota
	Lasso
	Ridge
	ElasticNet
)

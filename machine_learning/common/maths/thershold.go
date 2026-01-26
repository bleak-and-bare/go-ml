package maths

type Threshold struct {
	CostEps   float32 // epsilon for relative convergence criterion of the cost function
	GradEps   float32
	MaxEpochs int
	MinEphocs int
}

func DefThreshold() Threshold {
	return Threshold{
		CostEps:   1e-7,
		GradEps:   1e-5,
		MinEphocs: 5,
		MaxEpochs: 100_000,
	}
}

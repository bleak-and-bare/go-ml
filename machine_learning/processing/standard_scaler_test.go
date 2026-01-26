package processing

import (
	"slices"
	"testing"
)

func TestStandardScaler_Fit(t *testing.T) {
	tests := []struct {
		name  string
		it    []float32
		mean  float32
		stdev float32
	}{
		{
			"Set with no variance",
			[]float32{1.0, 1.0, 1.0},
			1.0,
			0.0,
		}, {
			"All zeros",
			[]float32{0.0, 0.0, 0.0},
			0.0,
			0.0,
		}, {
			"Set with correct value",
			[]float32{1.0, 2.0},
			1.5,
			0.5,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var s StandardScaler[float32]
			s.Fit(slices.Values(tt.it))

			if s.mean != tt.mean {
				t.Errorf("Wrong mean : %.3f. Expected : %.3f", s.mean, tt.mean)
			}

			if s.stdev != tt.stdev {
				t.Errorf("Wrong stdev : %.3f. Expected : %.3f", s.stdev, tt.stdev)
			}
		})
	}
}

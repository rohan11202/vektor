package vektor

import (
	"math"
	"slices"
)

// Quantizer maps unit vectors to int8 codes with one scale per dimension,
// cutting vector memory by 4x compared to float32.
type Quantizer struct {
	scale []float32
}

// TrainQuantizer sets each dimension's range from the given quantile of
// absolute values in sample. A quantile just under 1 (e.g. 0.999) clips rare
// outliers so the common range keeps more resolution.
func TrainQuantizer(sample [][]float32, quantile float64) *Quantizer {
	if len(sample) == 0 {
		panic("vektor: empty quantizer sample")
	}
	if quantile <= 0 || quantile > 1 {
		panic("vektor: quantile must be in (0, 1]")
	}
	dim := len(sample[0])
	cols := make([][]float32, dim)
	for _, v := range sample {
		checkDim(dim, v)
		for d, x := range normalize(v) {
			cols[d] = append(cols[d], float32(math.Abs(float64(x))))
		}
	}
	scale := make([]float32, dim)
	for d, col := range cols {
		slices.Sort(col)
		m := col[int(quantile*float64(len(col)-1))]
		if m == 0 {
			m = 1
		}
		scale[d] = m / 127
	}
	return &Quantizer{scale: scale}
}

func (q *Quantizer) Dim() int { return len(q.scale) }

func (q *Quantizer) Encode(v []float32, dst []int8) {
	for i, x := range v {
		c := math.Round(float64(x / q.scale[i]))
		dst[i] = int8(max(-127, min(127, c)))
	}
}

func (q *Quantizer) Decode(c []int8, dst []float32) {
	for i, x := range c {
		dst[i] = float32(x) * q.scale[i]
	}
}

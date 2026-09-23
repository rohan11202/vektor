package vektor

import "math"

// dot is unrolled by four; this is the hot loop for every index.
func dot(a, b []float32) float32 {
	var s0, s1, s2, s3 float32
	n := len(a) &^ 3
	for i := 0; i < n; i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for i := n; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return s0 + s1 + s2 + s3
}

// normalize returns a unit-length copy of v, so cosine similarity becomes a dot product.
func normalize(v []float32) []float32 {
	out := make([]float32, len(v))
	var n float64
	for _, x := range v {
		n += float64(x) * float64(x)
	}
	if n == 0 {
		return out
	}
	inv := float32(1 / math.Sqrt(n))
	for i, x := range v {
		out[i] = x * inv
	}
	return out
}

// Cosine returns the cosine similarity of a and b.
func Cosine(a, b []float32) float32 {
	return dot(normalize(a), normalize(b))
}

func checkDim(want int, v []float32) {
	if len(v) != want {
		panic("vektor: vector has wrong dimension")
	}
}

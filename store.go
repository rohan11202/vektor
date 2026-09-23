package vektor

// vectorStore holds unit vectors for the graph. Scores are cosine similarity.
type vectorStore interface {
	add(v []float32)
	vector(id int) []float32
	scorer(q []float32) func(id int) float32
}

type floatStore struct {
	dim  int
	data []float32
}

func (s *floatStore) add(v []float32) { s.data = append(s.data, v...) }

func (s *floatStore) vector(id int) []float32 {
	return s.data[id*s.dim : (id+1)*s.dim]
}

func (s *floatStore) scorer(q []float32) func(int) float32 {
	return func(id int) float32 { return dot(q, s.vector(id)) }
}

type int8Store struct {
	dim   int
	q     *Quantizer
	codes []int8
}

func (s *int8Store) add(v []float32) {
	n := len(s.codes)
	s.codes = append(s.codes, make([]int8, s.dim)...)
	s.q.Encode(v, s.codes[n:])
}

func (s *int8Store) vector(id int) []float32 {
	out := make([]float32, s.dim)
	s.q.Decode(s.codes[id*s.dim:(id+1)*s.dim], out)
	return out
}

// scorer folds the per-dimension scale into the query once, so each
// comparison is a plain dot product against the raw codes.
func (s *int8Store) scorer(q []float32) func(int) float32 {
	qs := make([]float32, s.dim)
	for i, x := range q {
		qs[i] = x * s.q.scale[i]
	}
	return func(id int) float32 {
		c := s.codes[id*s.dim : (id+1)*s.dim]
		var sum float32
		for i, x := range c {
			sum += qs[i] * float32(x)
		}
		return sum
	}
}

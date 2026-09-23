package vektor

// Flat is an exact index. It scans every vector and serves as recall ground truth.
type Flat struct {
	dim  int
	data []float32
}

func NewFlat(dim int) *Flat {
	return &Flat{dim: dim}
}

// Add inserts v and returns its id.
func (f *Flat) Add(v []float32) int {
	checkDim(f.dim, v)
	f.data = append(f.data, normalize(v)...)
	return f.Len() - 1
}

func (f *Flat) Len() int {
	return len(f.data) / f.dim
}

// Search returns the k most similar vectors to q, best first. Safe for concurrent use once loading is done.
func (f *Flat) Search(q []float32, k int) []Result {
	checkDim(f.dim, q)
	q = normalize(q)
	top := newTopK(k)
	for i, off := 0, 0; off < len(f.data); i, off = i+1, off+f.dim {
		top.offer(Result{ID: i, Score: dot(q, f.data[off:off+f.dim])})
	}
	return top.sorted()
}

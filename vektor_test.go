package vektor

import (
	"bytes"
	"math"
	"math/rand"
	"slices"
	"testing"
)

func clustered(n, dim int, rng *rand.Rand) [][]float32 {
	centers := make([][]float32, 20)
	for i := range centers {
		centers[i] = make([]float32, dim)
		for d := range centers[i] {
			centers[i][d] = float32(rng.NormFloat64())
		}
	}
	out := make([][]float32, n)
	for i := range out {
		c := centers[rng.Intn(len(centers))]
		v := make([]float32, dim)
		for d := range v {
			v[d] = c[d] + 0.5*float32(rng.NormFloat64())
		}
		out[i] = v
	}
	return out
}

func recall(got, want []Result) float64 {
	ids := map[int]bool{}
	for _, r := range want {
		ids[r.ID] = true
	}
	hit := 0
	for _, r := range got {
		if ids[r.ID] {
			hit++
		}
	}
	return float64(hit) / float64(len(want))
}

func TestTopK(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	scores := make([]float32, 500)
	top := newTopK(10)
	for i := range scores {
		scores[i] = rng.Float32()
		top.offer(Result{ID: i, Score: scores[i]})
	}
	got := top.sorted()
	slices.Sort(scores)
	slices.Reverse(scores)
	for i, r := range got {
		if r.Score != scores[i] {
			t.Fatalf("rank %d: got %v, want %v", i, r.Score, scores[i])
		}
	}
}

func TestFlatFindsItself(t *testing.T) {
	data := clustered(1000, 16, rand.New(rand.NewSource(1)))
	f := NewFlat(16)
	for _, v := range data {
		f.Add(v)
	}
	for _, i := range []int{0, 17, 999} {
		if r := f.Search(data[i], 1); r[0].ID != i {
			t.Fatalf("query %d: top hit %d", i, r[0].ID)
		}
	}
}

func buildPair(t *testing.T, q *Quantizer) (*HNSW, *Flat, [][]float32) {
	t.Helper()
	rng := rand.New(rand.NewSource(3))
	data := clustered(5000, 32, rng)
	f := NewFlat(32)
	var h *HNSW
	if q == nil {
		h = NewHNSW(32, DefaultConfig())
	} else {
		h = NewQuantizedHNSW(32, DefaultConfig(), q)
	}
	for _, v := range data {
		f.Add(v)
		h.Add(v)
	}
	return h, f, clustered(100, 32, rng)
}

func meanRecall(h *HNSW, f *Flat, queries [][]float32, ef int) float64 {
	var sum float64
	for _, q := range queries {
		sum += recall(h.SearchEf(q, 10, ef), f.Search(q, 10))
	}
	return sum / float64(len(queries))
}

func TestHNSWRecall(t *testing.T) {
	h, f, queries := buildPair(t, nil)
	if r := meanRecall(h, f, queries, 100); r < 0.95 {
		t.Fatalf("recall@10 = %.3f, want >= 0.95", r)
	}
}

func TestQuantizedRecall(t *testing.T) {
	sample := clustered(2000, 32, rand.New(rand.NewSource(3)))
	h, f, queries := buildPair(t, TrainQuantizer(sample, 0.999))
	if r := meanRecall(h, f, queries, 100); r < 0.85 {
		t.Fatalf("int8 recall@10 = %.3f, want >= 0.85", r)
	}
}

func TestQuantizerRoundTrip(t *testing.T) {
	data := clustered(500, 24, rand.New(rand.NewSource(5)))
	q := TrainQuantizer(data, 1)
	code := make([]int8, 24)
	back := make([]float32, 24)
	for _, v := range data {
		u := normalize(v)
		q.Encode(u, code)
		q.Decode(code, back)
		for d := range u {
			if err := math.Abs(float64(u[d] - back[d])); err > float64(q.scale[d])/2+1e-6 {
				t.Fatalf("dim %d: error %g above half a step", d, err)
			}
		}
	}
}

func TestSaveLoad(t *testing.T) {
	sample := clustered(1000, 32, rand.New(rand.NewSource(3)))
	for _, q := range []*Quantizer{nil, TrainQuantizer(sample, 0.999)} {
		h, _, queries := buildPair(t, q)
		var buf bytes.Buffer
		if err := h.Save(&buf); err != nil {
			t.Fatal(err)
		}
		loaded, err := Load(&buf)
		if err != nil {
			t.Fatal(err)
		}
		if loaded.Len() != h.Len() {
			t.Fatalf("len %d, want %d", loaded.Len(), h.Len())
		}
		for _, qv := range queries[:20] {
			if !slices.Equal(h.Search(qv, 10), loaded.Search(qv, 10)) {
				t.Fatal("results differ after reload")
			}
		}
	}
}

func TestLoadRejectsGarbage(t *testing.T) {
	if _, err := Load(bytes.NewReader([]byte("nope, not an index"))); err == nil {
		t.Fatal("expected error")
	}
}

package vektor

// Result is a search hit. Higher Score means more similar.
type Result struct {
	ID    int
	Score float32
}

// minHeap orders results by ascending score.
type minHeap []Result

func (h *minHeap) push(r Result) {
	*h = append(*h, r)
	s := *h
	i := len(s) - 1
	for i > 0 {
		p := (i - 1) / 2
		if s[p].Score <= s[i].Score {
			break
		}
		s[p], s[i] = s[i], s[p]
		i = p
	}
}

func (h *minHeap) pop() Result {
	s := *h
	top := s[0]
	last := len(s) - 1
	s[0] = s[last]
	*h = s[:last]
	h.down(0)
	return top
}

func (h minHeap) down(i int) {
	n := len(h)
	for {
		l := 2*i + 1
		if l >= n {
			return
		}
		m := l
		if r := l + 1; r < n && h[r].Score < h[l].Score {
			m = r
		}
		if h[i].Score <= h[m].Score {
			return
		}
		h[i], h[m] = h[m], h[i]
		i = m
	}
}

// topK keeps the k best results seen so far. The root is the weakest kept
// result, so rejecting a candidate costs one comparison.
type topK struct {
	k int
	h minHeap
}

func newTopK(k int) *topK {
	return &topK{k: k, h: make(minHeap, 0, k+1)}
}

func (t *topK) full() bool     { return len(t.h) >= t.k }
func (t *topK) worst() float32 { return t.h[0].Score }

// offer adds r if there is room or it beats the current worst. It reports whether r was kept.
func (t *topK) offer(r Result) bool {
	if len(t.h) < t.k {
		t.h.push(r)
		return true
	}
	if r.Score <= t.h[0].Score {
		return false
	}
	t.h[0] = r
	t.h.down(0)
	return true
}

// sorted drains the heap and returns results best first.
func (t *topK) sorted() []Result {
	out := make([]Result, len(t.h))
	for i := len(out) - 1; i >= 0; i-- {
		out[i] = t.h.pop()
	}
	return out
}

package vektor

import (
	"math"
	"math/rand"
	"slices"
	"sync"
)

// Config holds the HNSW build and search parameters.
type Config struct {
	M              int // links per node on upper layers; layer 0 allows 2*M
	EfConstruction int // candidate list size while inserting
	EfSearch       int // default candidate list size while searching
	Seed           int64
}

func DefaultConfig() Config {
	return Config{M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1}
}

// HNSW is a Hierarchical Navigable Small World graph (Malkov & Yashunin)
// over cosine similarity. Searches may run concurrently; Add is exclusive.
type HNSW struct {
	mu        sync.RWMutex
	cfg       Config
	dim       int
	store     vectorStore
	quant     *Quantizer
	links     [][][]uint32 // node -> layer -> neighbor ids
	entry     int
	maxLayer  int
	levelMult float64
	rng       *rand.Rand
	visited   sync.Pool
}

func NewHNSW(dim int, cfg Config) *HNSW {
	return newHNSW(dim, cfg, nil)
}

// NewQuantizedHNSW stores vectors as int8 codes from q instead of float32.
func NewQuantizedHNSW(dim int, cfg Config, q *Quantizer) *HNSW {
	if q.Dim() != dim {
		panic("vektor: quantizer dimension mismatch")
	}
	return newHNSW(dim, cfg, q)
}

func newHNSW(dim int, cfg Config, q *Quantizer) *HNSW {
	def := DefaultConfig()
	if cfg.M < 2 {
		cfg.M = def.M
	}
	if cfg.EfConstruction < cfg.M {
		cfg.EfConstruction = max(def.EfConstruction, cfg.M)
	}
	if cfg.EfSearch <= 0 {
		cfg.EfSearch = def.EfSearch
	}
	h := &HNSW{
		cfg:       cfg,
		dim:       dim,
		quant:     q,
		entry:     -1,
		levelMult: 1 / math.Log(float64(cfg.M)),
		rng:       rand.New(rand.NewSource(cfg.Seed)),
	}
	if q == nil {
		h.store = &floatStore{dim: dim}
	} else {
		h.store = &int8Store{dim: dim, q: q}
	}
	return h
}

func (h *HNSW) Config() Config { return h.cfg }

func (h *HNSW) Len() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.links)
}

// randomLayer draws from an exponential distribution, so each layer holds about 1/M of the one below.
func (h *HNSW) randomLayer() int {
	return int(-math.Log(1-h.rng.Float64()) * h.levelMult)
}

func (h *HNSW) maxLinks(layer int) int {
	if layer == 0 {
		return 2 * h.cfg.M
	}
	return h.cfg.M
}

// Add inserts v and returns its id. Ids follow insertion order.
func (h *HNSW) Add(v []float32) int {
	checkDim(h.dim, v)
	h.mu.Lock()
	defer h.mu.Unlock()

	id := len(h.links)
	u := normalize(v)
	h.store.add(u)
	layer := h.randomLayer()
	h.links = append(h.links, make([][]uint32, layer+1))
	if h.entry < 0 {
		h.entry, h.maxLayer = id, layer
		return id
	}

	score := h.store.scorer(u)
	ep := Result{ID: h.entry, Score: score(h.entry)}
	for l := h.maxLayer; l > layer; l-- {
		ep = h.greedy(score, ep, l)
	}
	for l := min(layer, h.maxLayer); l >= 0; l-- {
		cands := h.searchLayer(score, ep, h.cfg.EfConstruction, l)
		nbrs := h.selectNeighbors(cands, h.cfg.M)
		h.links[id][l] = nbrs
		for _, nb := range nbrs {
			h.connect(int(nb), uint32(id), l)
		}
		ep = cands[0]
	}
	if layer > h.maxLayer {
		h.entry, h.maxLayer = id, layer
	}
	return id
}

// connect adds a back-link node->id and prunes node's list if it overflows.
func (h *HNSW) connect(node int, id uint32, layer int) {
	nbrs := append(h.links[node][layer], id)
	if len(nbrs) > h.maxLinks(layer) {
		score := h.store.scorer(h.store.vector(node))
		cands := make([]Result, len(nbrs))
		for i, n := range nbrs {
			cands[i] = Result{ID: int(n), Score: score(int(n))}
		}
		slices.SortFunc(cands, func(a, b Result) int {
			switch {
			case a.Score > b.Score:
				return -1
			case a.Score < b.Score:
				return 1
			}
			return 0
		})
		nbrs = h.selectNeighbors(cands, h.maxLinks(layer))
	}
	h.links[node][layer] = nbrs
}

// selectNeighbors keeps a candidate only if it is closer to the base node than
// to every neighbor already kept, which spreads links across directions.
// Leftover slots go to the best skipped candidates. cands must be sorted best first.
func (h *HNSW) selectNeighbors(cands []Result, m int) []uint32 {
	kept := make([]uint32, 0, min(m, len(cands)))
	if len(cands) <= m {
		for _, c := range cands {
			kept = append(kept, uint32(c.ID))
		}
		return kept
	}
	var skipped []Result
	for _, c := range cands {
		if len(kept) == m {
			break
		}
		score := h.store.scorer(h.store.vector(c.ID))
		diverse := true
		for _, k := range kept {
			if score(int(k)) > c.Score {
				diverse = false
				break
			}
		}
		if diverse {
			kept = append(kept, uint32(c.ID))
		} else {
			skipped = append(skipped, c)
		}
	}
	for _, c := range skipped {
		if len(kept) == m {
			break
		}
		kept = append(kept, uint32(c.ID))
	}
	return kept
}

// greedy moves to a better neighbor on layer l until none is left.
func (h *HNSW) greedy(score func(int) float32, ep Result, l int) Result {
	for changed := true; changed; {
		changed = false
		for _, nb := range h.links[ep.ID][l] {
			if s := score(int(nb)); s > ep.Score {
				ep = Result{ID: int(nb), Score: s}
				changed = true
			}
		}
	}
	return ep
}

// searchLayer is a best-first search on layer l that keeps the ef best nodes seen.
func (h *HNSW) searchLayer(score func(int) float32, ep Result, ef, l int) []Result {
	vis := h.getVisited()
	defer h.visited.Put(vis)
	vis.visit(ep.ID)

	// Scores are negated so the min-heap pops the most similar candidate first.
	cands := minHeap{{ID: ep.ID, Score: -ep.Score}}
	top := newTopK(ef)
	top.offer(ep)
	for len(cands) > 0 {
		c := cands.pop()
		if top.full() && -c.Score < top.worst() {
			break
		}
		for _, nb := range h.links[c.ID][l] {
			n := int(nb)
			if vis.visit(n) {
				continue
			}
			s := score(n)
			if top.offer(Result{ID: n, Score: s}) {
				cands.push(Result{ID: n, Score: -s})
			}
		}
	}
	return top.sorted()
}

// Search returns the k nearest neighbors of q, best first, using the configured EfSearch.
func (h *HNSW) Search(q []float32, k int) []Result {
	return h.SearchEf(q, k, h.cfg.EfSearch)
}

// SearchEf is Search with an explicit candidate list size. Larger ef gives
// better recall and slower queries.
func (h *HNSW) SearchEf(q []float32, k, ef int) []Result {
	checkDim(h.dim, q)
	h.mu.RLock()
	defer h.mu.RUnlock()
	if h.entry < 0 {
		return nil
	}
	score := h.store.scorer(normalize(q))
	ep := Result{ID: h.entry, Score: score(h.entry)}
	for l := h.maxLayer; l > 0; l-- {
		ep = h.greedy(score, ep, l)
	}
	res := h.searchLayer(score, ep, max(ef, k), 0)
	if len(res) > k {
		res = res[:k]
	}
	return res
}

// visitedSet marks nodes with a generation counter, so reuse needs no clearing.
type visitedSet struct {
	marks []uint32
	gen   uint32
}

func (h *HNSW) getVisited() *visitedSet {
	v, _ := h.visited.Get().(*visitedSet)
	if v == nil {
		v = &visitedSet{}
	}
	if n := len(h.links); len(v.marks) < n {
		v.marks = make([]uint32, n+n/2)
		v.gen = 0
	}
	v.gen++
	if v.gen == 0 {
		clear(v.marks)
		v.gen = 1
	}
	return v
}

// visit marks id and reports whether it was already seen.
func (v *visitedSet) visit(id int) bool {
	if v.marks[id] == v.gen {
		return true
	}
	v.marks[id] = v.gen
	return false
}

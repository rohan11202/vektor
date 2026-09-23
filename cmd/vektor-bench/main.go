// Command vektor-bench measures recall@k and QPS of vektor's HNSW index,
// float32 and int8, against exact search and optionally Elasticsearch.
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"text/tabwriter"
	"time"

	"github.com/rohan11202/vektor"
)

func main() {
	var (
		n        = flag.Int("n", 100000, "base vectors to generate (ignored with -base)")
		dim      = flag.Int("dim", 128, "vector dimension for generated data")
		nq       = flag.Int("queries", 1000, "number of queries")
		k        = flag.Int("k", 10, "neighbors per query")
		base     = flag.String("base", "", "base vectors in .fvecs format")
		query    = flag.String("query", "", "query vectors in .fvecs format; default holds out the tail of -base")
		m        = flag.Int("m", 16, "HNSW M")
		efc      = flag.Int("efc", 200, "HNSW efConstruction")
		efFlag   = flag.String("ef", "10,20,40,80,160,320", "efSearch / num_candidates values to sweep")
		int8Run  = flag.Bool("int8", true, "also benchmark the int8 quantized index")
		esURL    = flag.String("es", "", "Elasticsearch URL, e.g. http://localhost:9200; empty skips ES")
		esType   = flag.String("es-type", "hnsw", "dense_vector index_options type: hnsw, int8_hnsw, bbq_hnsw")
		savePath = flag.String("save", "", "write the float32 index to this file and check it reloads")
		seed     = flag.Int64("seed", 42, "random seed")
	)
	flag.Parse()

	efs, err := parseInts(*efFlag)
	if err != nil {
		log.Fatalf("-ef: %v", err)
	}
	data, queries, err := loadData(*base, *query, *n, *dim, *nq, *seed)
	if err != nil {
		log.Fatal(err)
	}
	d := len(data[0])
	fmt.Printf("dataset: %d base, %d queries, dim %d, k %d\n", len(data), len(queries), d, *k)

	start := time.Now()
	gt := groundTruth(data, queries, *k)
	fmt.Printf("ground truth (exact, %d threads): %v\n\n", runtime.NumCPU(), time.Since(start).Round(time.Millisecond))

	cfg := vektor.Config{M: *m, EfConstruction: *efc, EfSearch: efs[0], Seed: *seed}
	var rows []row

	idx := vektor.NewHNSW(d, cfg)
	build("hnsw-f32", idx, data)
	rows = append(rows, sweep("hnsw-f32", hnswSearcher(idx), queries, gt, efs, *k)...)

	if *int8Run {
		sample := data[:min(len(data), 20000)]
		qidx := vektor.NewQuantizedHNSW(d, cfg, vektor.TrainQuantizer(sample, 0.999))
		build("hnsw-int8", qidx, data)
		rows = append(rows, sweep("hnsw-int8", hnswSearcher(qidx), queries, gt, efs, *k)...)
	}

	if *savePath != "" {
		if err := saveAndCheck(idx, *savePath, queries[0], *k); err != nil {
			log.Fatal(err)
		}
	}

	if *esURL != "" {
		es := newESClient(*esURL, "vektor-bench")
		start := time.Now()
		if err := es.createIndex(d, *m, *efc, *esType); err != nil {
			log.Fatal(err)
		}
		if err := es.bulk(data, 1000); err != nil {
			log.Fatal(err)
		}
		if err := es.finish(); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("build es-%s: %v (bulk + force merge)\n", *esType, time.Since(start).Round(time.Millisecond))
		search := func(q []float32, k, ef int) []int {
			ids, err := es.knn(q, k, max(ef, k))
			if err != nil {
				log.Fatal(err)
			}
			return ids
		}
		rows = append(rows, sweep("es-"+*esType, search, queries, gt, efs, *k)...)
	}

	fmt.Println()
	printTable(rows, *k)
}

type searcher func(q []float32, k, ef int) []int

type row struct {
	engine   string
	ef       int
	recall   float64
	qps      float64
	p50, p99 time.Duration
}

func hnswSearcher(h *vektor.HNSW) searcher {
	return func(q []float32, k, ef int) []int {
		res := h.SearchEf(q, k, ef)
		ids := make([]int, len(res))
		for i, r := range res {
			ids[i] = r.ID
		}
		return ids
	}
}

func build(name string, h *vektor.HNSW, data [][]float32) {
	start := time.Now()
	for _, v := range data {
		h.Add(v)
	}
	el := time.Since(start)
	fmt.Printf("build %s: %v (%.0f vectors/s)\n", name, el.Round(time.Millisecond), float64(len(data))/el.Seconds())
}

// sweep runs every query single-threaded at each ef, so QPS is 1/mean latency.
func sweep(name string, search searcher, queries [][]float32, gt [][]int, efs []int, k int) []row {
	var rows []row
	for _, ef := range efs {
		for _, q := range queries[:min(len(queries), 100)] {
			search(q, k, ef) // warm caches before timing
		}
		lat := make([]time.Duration, len(queries))
		var hits int
		start := time.Now()
		for i, q := range queries {
			t := time.Now()
			ids := search(q, k, ef)
			lat[i] = time.Since(t)
			hits += overlap(ids, gt[i])
		}
		total := time.Since(start)
		slices.Sort(lat)
		rows = append(rows, row{
			engine: name,
			ef:     ef,
			recall: float64(hits) / float64(len(queries)*k),
			qps:    float64(len(queries)) / total.Seconds(),
			p50:    lat[len(lat)/2],
			p99:    lat[len(lat)*99/100],
		})
	}
	return rows
}

func overlap(got, want []int) int {
	n := 0
	for _, g := range got {
		if slices.Contains(want, g) {
			n++
		}
	}
	return n
}

func groundTruth(data, queries [][]float32, k int) [][]int {
	flat := vektor.NewFlat(len(data[0]))
	for _, v := range data {
		flat.Add(v)
	}
	gt := make([][]int, len(queries))
	jobs := make(chan int)
	var wg sync.WaitGroup
	for w := 0; w < runtime.NumCPU(); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				for _, r := range flat.Search(queries[i], k) {
					gt[i] = append(gt[i], r.ID)
				}
			}
		}()
	}
	for i := range queries {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	return gt
}

func loadData(base, query string, n, dim, nq int, seed int64) ([][]float32, [][]float32, error) {
	if base == "" {
		rng := rand.New(rand.NewSource(seed))
		all := synthetic(n+nq, dim, 100, rng)
		return all[:n], all[n:], nil
	}
	data, err := readFvecs(base, 0)
	if err != nil {
		return nil, nil, err
	}
	if query != "" {
		queries, err := readFvecs(query, nq)
		return data, queries, err
	}
	if len(data) <= nq {
		return nil, nil, fmt.Errorf("%s: need more than %d vectors to hold out queries", base, nq)
	}
	return data[:len(data)-nq], data[len(data)-nq:], nil
}

func saveAndCheck(h *vektor.HNSW, path string, q []float32, k int) error {
	if err := h.SaveFile(path); err != nil {
		return err
	}
	st, err := os.Stat(path)
	if err != nil {
		return err
	}
	loaded, err := vektor.LoadFile(path)
	if err != nil {
		return err
	}
	if !slices.Equal(h.Search(q, k), loaded.Search(q, k)) {
		return fmt.Errorf("%s: reloaded index returns different results", path)
	}
	fmt.Printf("saved %s (%.1f MB), reload ok\n", path, float64(st.Size())/(1<<20))
	return nil
}

func printTable(rows []row, k int) {
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintf(w, "engine\tef\trecall@%d\tQPS\tp50\tp99\t\n", k)
	for _, r := range rows {
		fmt.Fprintf(w, "%s\t%d\t%.4f\t%.0f\t%s\t%s\t\n", r.engine, r.ef, r.recall, r.qps, fmtDur(r.p50), fmtDur(r.p99))
	}
	w.Flush()
}

func fmtDur(d time.Duration) string {
	return fmt.Sprintf("%.3fms", float64(d.Microseconds())/1000)
}

func parseInts(s string) ([]int, error) {
	var out []int
	for _, f := range strings.Split(s, ",") {
		v, err := strconv.Atoi(strings.TrimSpace(f))
		if err != nil || v <= 0 {
			return nil, fmt.Errorf("bad value %q", f)
		}
		out = append(out, v)
	}
	return out, nil
}

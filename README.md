# vektor

A vector search engine written from scratch in Go. No dependencies outside the standard library.

- **HNSW** index (Malkov & Yashunin) with tunable `M`, `efConstruction` and `efSearch`
- **Exact search** baseline (cosine similarity, bounded min-heap top-k), used as recall ground truth
- **int8 scalar quantization** with per-dimension scales, 4x less vector memory than float32
- **Disk persistence** in a small binary format, written atomically
- **Benchmark tool** that measures recall@k, QPS and latency, with an optional run against Elasticsearch `dense_vector`

## Usage

```go
idx := vektor.NewHNSW(128, vektor.Config{M: 16, EfConstruction: 200, EfSearch: 64})
for _, v := range vectors {
    idx.Add(v) // ids follow insertion order
}

hits := idx.Search(query, 10)       // []vektor.Result{ID, Score}, best first
hits = idx.SearchEf(query, 10, 200) // bigger ef = better recall, slower

idx.SaveFile("vectors.idx")
idx, err := vektor.LoadFile("vectors.idx")
```

Quantized index:

```go
q := vektor.TrainQuantizer(sample, 0.999) // clip the top 0.1% per dimension
idx := vektor.NewQuantizedHNSW(128, vektor.DefaultConfig(), q)
```

Similarity is always cosine. Vectors are normalized on insert, so scoring is a plain dot product.
Searches can run concurrently. `Add` takes an exclusive lock.

## How it works

| Piece | File | Notes |
|---|---|---|
| Graph build and search | `hnsw.go` | layer drawn from an exponential distribution, greedy descent on upper layers, best-first search with `ef` candidates on layer 0 |
| Neighbor selection | `hnsw.go` | keeps a candidate only if it is closer to the node than to any neighbor already kept; stops links bunching in one direction |
| Top-k | `heap.go` | min-heap of size k; root is the weakest result, so most candidates are rejected with one compare |
| Visited set | `hnsw.go` | generation counter per search, pooled; no map or clearing per query |
| int8 | `quant.go`, `store.go` | query is multiplied by the scales once, then each compare is float32 x int8 over the codes |
| File format | `persist.go` | header, vectors or codes, then one flat stream of links |

## Benchmark

```sh
go run ./cmd/vektor-bench -n 50000 -queries 500
go run ./cmd/vektor-bench -base sift_base.fvecs -query sift_query.fvecs
go run ./cmd/vektor-bench -es http://localhost:9200 -es-type int8_hnsw
```

Ground truth comes from the exact index. Each engine runs every query single-threaded
after a warm-up pass, so QPS = 1 / mean latency.

Results on 50,000 clustered synthetic vectors, 128 dims, 500 queries, `M=16`,
`efConstruction=200`, k=10. Laptop i7-1355U. Elasticsearch 9.4.0 in Docker, 1 shard, force-merged to 1 segment, same `M` and `ef_construction`.

| engine | ef | recall@10 | QPS | p50 | p99 |
|---|---:|---:|---:|---:|---:|
| hnsw-f32 | 10 | 0.8186 | 20050 | 0.049 ms | 0.072 ms |
| hnsw-f32 | 20 | 0.9370 | 14177 | 0.070 ms | 0.094 ms |
| hnsw-f32 | 40 | 0.9912 | 8998 | 0.103 ms | 0.287 ms |
| hnsw-f32 | 80 | 0.9992 | 6890 | 0.142 ms | 0.378 ms |
| hnsw-f32 | 160 | 1.0000 | 4998 | 0.196 ms | 0.354 ms |
| hnsw-int8 | 10 | 0.8040 | 27031 | 0.035 ms | 0.063 ms |
| hnsw-int8 | 40 | 0.9424 | 14254 | 0.068 ms | 0.090 ms |
| hnsw-int8 | 160 | 0.9478 | 7575 | 0.130 ms | 0.179 ms |
| es-hnsw | 10 | 0.8238 | 503 | 1.947 ms | 3.295 ms |
| es-hnsw | 40 | 0.9824 | 735 | 1.306 ms | 2.182 ms |
| es-hnsw | 160 | 0.9968 | 802 | 1.164 ms | 2.299 ms |

Notes:

- Recall tracks Elasticsearch closely at every `ef`, which is a good check that the graph is built right.
- ES latency includes the HTTP round trip and JSON encoding of the query vector. vektor runs in process.
  Compare recall curves, not raw QPS.
- int8 stops near 0.95 recall because there is no float rescoring step. The graph finds the right
  area, but the rounding error reorders close neighbors.
- Build is single-threaded: about 2,800 vectors/s for float32.

## Tests

```sh
go test ./...
```

Covers top-k order, exact search, HNSW and int8 recall floors, quantizer error bounds and save/load round trips.

## License

MIT

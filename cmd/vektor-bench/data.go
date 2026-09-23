package main

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"os"
)

// readFvecs reads the TEXMEX .fvecs format: each record is an int32 dimension followed by that many float32s.
func readFvecs(path string, limit int) ([][]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 1<<20)
	var out [][]float32
	for limit <= 0 || len(out) < limit {
		var d int32
		if err := binary.Read(r, binary.LittleEndian, &d); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, err
		}
		if d <= 0 || d > 1<<16 {
			return nil, fmt.Errorf("%s: bad dimension %d", path, d)
		}
		v := make([]float32, d)
		if err := binary.Read(r, binary.LittleEndian, v); err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	return out, nil
}

// synthetic draws points around random cluster centers. Uniform noise has no
// structure and makes every ANN index look bad; real embeddings cluster.
func synthetic(n, dim, clusters int, rng *rand.Rand) [][]float32 {
	centers := make([][]float32, clusters)
	for i := range centers {
		centers[i] = make([]float32, dim)
		for d := range centers[i] {
			centers[i][d] = float32(rng.NormFloat64())
		}
	}
	out := make([][]float32, n)
	for i := range out {
		c := centers[rng.Intn(clusters)]
		v := make([]float32, dim)
		for d := range v {
			v[d] = c[d] + 0.6*float32(rng.NormFloat64())
		}
		out[i] = v
	}
	return out
}

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
)

// esClient is a minimal Elasticsearch REST client for dense_vector kNN.
type esClient struct {
	url   string
	index string
	http  *http.Client
}

func newESClient(url, index string) *esClient {
	return &esClient{url: strings.TrimRight(url, "/"), index: index, http: &http.Client{}}
}

func (c *esClient) do(method, path, contentType string, body []byte, out any) error {
	req, err := http.NewRequest(method, c.url+path, bytes.NewReader(body))
	if err != nil {
		return err
	}
	if body != nil {
		req.Header.Set("Content-Type", contentType)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if resp.StatusCode >= 300 {
		return fmt.Errorf("%s %s: %s: %.300s", method, path, resp.Status, data)
	}
	if out != nil {
		return json.Unmarshal(data, out)
	}
	return nil
}

func (c *esClient) json(method, path string, body, out any) error {
	var b []byte
	if body != nil {
		var err error
		if b, err = json.Marshal(body); err != nil {
			return err
		}
	}
	return c.do(method, path, "application/json", b, out)
}

// createIndex drops any old index and creates one shard with no replicas, so
// the comparison is one HNSW graph against one HNSW graph.
func (c *esClient) createIndex(dim, m, efc int, indexType string) error {
	c.json(http.MethodDelete, "/"+c.index+"?ignore_unavailable=true", nil, nil)
	return c.json(http.MethodPut, "/"+c.index, map[string]any{
		"settings": map[string]any{
			"number_of_shards":   1,
			"number_of_replicas": 0,
			"refresh_interval":   "-1",
		},
		"mappings": map[string]any{
			"properties": map[string]any{
				"vec": map[string]any{
					"type":       "dense_vector",
					"dims":       dim,
					"index":      true,
					"similarity": "cosine",
					"index_options": map[string]any{
						"type":            indexType,
						"m":               m,
						"ef_construction": efc,
					},
				},
			},
		},
	}, nil)
}

func (c *esClient) bulk(vecs [][]float32, batch int) error {
	for start := 0; start < len(vecs); start += batch {
		var buf bytes.Buffer
		enc := json.NewEncoder(&buf)
		for i := start; i < min(start+batch, len(vecs)); i++ {
			enc.Encode(map[string]any{"index": map[string]any{"_id": strconv.Itoa(i)}})
			enc.Encode(map[string]any{"vec": vecs[i]})
		}
		var resp struct {
			Errors bool `json:"errors"`
		}
		if err := c.do(http.MethodPost, "/"+c.index+"/_bulk", "application/x-ndjson", buf.Bytes(), &resp); err != nil {
			return err
		}
		if resp.Errors {
			return fmt.Errorf("bulk batch at %d had item errors", start)
		}
	}
	return nil
}

// finish refreshes and force-merges to one segment. Otherwise ES searches one
// small graph per segment and looks slower than it is.
func (c *esClient) finish() error {
	if err := c.json(http.MethodPost, "/"+c.index+"/_refresh", nil, nil); err != nil {
		return err
	}
	return c.json(http.MethodPost, "/"+c.index+"/_forcemerge?max_num_segments=1", nil, nil)
}

func (c *esClient) knn(q []float32, k, numCandidates int) ([]int, error) {
	var resp struct {
		Hits struct {
			Hits []struct {
				ID string `json:"_id"`
			} `json:"hits"`
		} `json:"hits"`
	}
	err := c.json(http.MethodPost, "/"+c.index+"/_search", map[string]any{
		"knn": map[string]any{
			"field":          "vec",
			"query_vector":   q,
			"k":              k,
			"num_candidates": numCandidates,
		},
		"size":    k,
		"_source": false,
	}, &resp)
	if err != nil {
		return nil, err
	}
	ids := make([]int, len(resp.Hits.Hits))
	for i, h := range resp.Hits.Hits {
		if ids[i], err = strconv.Atoi(h.ID); err != nil {
			return nil, err
		}
	}
	return ids, nil
}

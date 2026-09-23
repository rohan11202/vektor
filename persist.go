package vektor

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

const (
	fileMagic     = "VKTR"
	formatVersion = 1
)

var le = binary.LittleEndian

// Save writes the index in a little-endian binary format:
// magic, header, quantizer scales (if any), vectors, then links.
func (h *HNSW) Save(w io.Writer) error {
	h.mu.RLock()
	defer h.mu.RUnlock()

	bw := bufio.NewWriter(w)
	quantized := int64(0)
	if h.quant != nil {
		quantized = 1
	}
	hdr := []int64{
		formatVersion, int64(h.dim), int64(h.cfg.M), int64(h.cfg.EfConstruction),
		int64(h.cfg.EfSearch), h.cfg.Seed, int64(len(h.links)), int64(h.entry),
		int64(h.maxLayer), quantized,
	}
	bw.WriteString(fileMagic)
	if err := binary.Write(bw, le, hdr); err != nil {
		return err
	}

	var err error
	switch s := h.store.(type) {
	case *floatStore:
		err = binary.Write(bw, le, s.data)
	case *int8Store:
		if err = binary.Write(bw, le, s.q.scale); err == nil {
			err = binary.Write(bw, le, s.codes)
		}
	}
	if err != nil {
		return err
	}

	// Links as one flat stream per node: layer count, then (count, ids...) per layer.
	var buf []uint32
	for _, layers := range h.links {
		buf = append(buf, uint32(len(layers)))
		for _, nbrs := range layers {
			buf = append(buf, uint32(len(nbrs)))
			buf = append(buf, nbrs...)
		}
	}
	if err := binary.Write(bw, le, uint64(len(buf))); err != nil {
		return err
	}
	if err := binary.Write(bw, le, buf); err != nil {
		return err
	}
	return bw.Flush()
}

// Load reads an index written by Save.
func Load(r io.Reader) (*HNSW, error) {
	br := bufio.NewReader(r)
	magic := make([]byte, len(fileMagic))
	if _, err := io.ReadFull(br, magic); err != nil {
		return nil, err
	}
	if string(magic) != fileMagic {
		return nil, errors.New("vektor: not an index file")
	}
	hdr := make([]int64, 10)
	if err := binary.Read(br, le, hdr); err != nil {
		return nil, err
	}
	if hdr[0] != formatVersion {
		return nil, fmt.Errorf("vektor: unsupported format version %d", hdr[0])
	}
	dim, n := int(hdr[1]), int(hdr[6])
	if dim <= 0 || n < 0 {
		return nil, errors.New("vektor: corrupt header")
	}
	cfg := Config{M: int(hdr[2]), EfConstruction: int(hdr[3]), EfSearch: int(hdr[4]), Seed: hdr[5]}

	var h *HNSW
	if hdr[9] == 1 {
		q := &Quantizer{scale: make([]float32, dim)}
		if err := binary.Read(br, le, q.scale); err != nil {
			return nil, err
		}
		h = newHNSW(dim, cfg, q)
		s := h.store.(*int8Store)
		s.codes = make([]int8, n*dim)
		if err := binary.Read(br, le, s.codes); err != nil {
			return nil, err
		}
	} else {
		h = newHNSW(dim, cfg, nil)
		s := h.store.(*floatStore)
		s.data = make([]float32, n*dim)
		if err := binary.Read(br, le, s.data); err != nil {
			return nil, err
		}
	}

	var size uint64
	if err := binary.Read(br, le, &size); err != nil {
		return nil, err
	}
	buf := make([]uint32, size)
	if err := binary.Read(br, le, buf); err != nil {
		return nil, err
	}
	h.links = make([][][]uint32, n)
	pos := 0
	next := func() (int, error) {
		if pos >= len(buf) {
			return 0, errors.New("vektor: truncated links")
		}
		pos++
		return int(buf[pos-1]), nil
	}
	for i := range h.links {
		nl, err := next()
		if err != nil {
			return nil, err
		}
		h.links[i] = make([][]uint32, nl)
		for l := range h.links[i] {
			c, err := next()
			if err != nil {
				return nil, err
			}
			if pos+c > len(buf) {
				return nil, errors.New("vektor: truncated links")
			}
			h.links[i][l] = append([]uint32(nil), buf[pos:pos+c]...)
			pos += c
		}
	}
	h.entry, h.maxLayer = int(hdr[7]), int(hdr[8])
	return h, nil
}

// SaveFile writes to a temp file and renames it, so a crash never leaves a half-written index.
func (h *HNSW) SaveFile(path string) error {
	tmp, err := os.CreateTemp(filepath.Dir(path), ".vektor-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if err := h.Save(tmp); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}

func LoadFile(path string) (*HNSW, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Load(f)
}

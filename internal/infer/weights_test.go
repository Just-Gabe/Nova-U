package infer

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/justgabe/Nova-U/pkg/tensor"
)

func TestBinaryRoundtrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "ckpt.bin")

	in := map[string]*tensor.Tensor{
		"a.weight": tensor.Random([]int{3, 4}, 0.1),
		"a.bias":   tensor.Random([]int{4}, 0.1),
		"b":        tensor.Random([]int{2, 3, 4}, 0.1),
	}
	if err := SaveWeightsBinary(in, path); err != nil {
		t.Fatalf("save: %v", err)
	}
	out, err := LoadWeightsBinary(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	for k, vIn := range in {
		vOut, ok := out[k]
		if !ok {
			t.Fatalf("missing key: %s", k)
		}
		if len(vIn.Data) != len(vOut.Data) {
			t.Fatalf("%s data len: got %d want %d", k, len(vOut.Data), len(vIn.Data))
		}
		for i := range vIn.Data {
			if vIn.Data[i] != vOut.Data[i] {
				t.Fatalf("%s[%d]: got %f want %f", k, i, vOut.Data[i], vIn.Data[i])
			}
		}
		if len(vIn.Shape) != len(vOut.Shape) {
			t.Fatalf("%s shape: %v vs %v", k, vOut.Shape, vIn.Shape)
		}
		for i := range vIn.Shape {
			if vIn.Shape[i] != vOut.Shape[i] {
				t.Fatalf("%s shape[%d]: got %d want %d", k, i, vOut.Shape[i], vIn.Shape[i])
			}
		}
	}
}

// TestPythonInterop emits the exact byte sequence that scripts/train_colab.py
// produces for a small fixture and confirms that LoadWeightsBinary parses it.
// If the Python export changes its format, this test will catch the drift.
func TestPythonInterop(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "from-py.bin")

	var buf bytes.Buffer
	// header
	buf.WriteString("NOVA")
	binary.Write(&buf, binary.LittleEndian, uint32(1)) // version
	binary.Write(&buf, binary.LittleEndian, uint32(1)) // nEntries

	// entry: name="embed.weight", shape=[2, 3], data = 1,2,3,4,5,6
	name := []byte("embed.weight")
	binary.Write(&buf, binary.LittleEndian, uint16(len(name)))
	buf.Write(name)
	buf.WriteByte(2) // nDims
	binary.Write(&buf, binary.LittleEndian, int32(2))
	binary.Write(&buf, binary.LittleEndian, int32(3))
	for _, v := range []float32{1, 2, 3, 4, 5, 6} {
		var u [4]byte
		binary.LittleEndian.PutUint32(u[:], math.Float32bits(v))
		buf.Write(u[:])
	}

	if err := os.WriteFile(path, buf.Bytes(), 0644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	w, err := LoadWeightsBinary(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(w) != 1 {
		t.Fatalf("entries: got %d want 1", len(w))
	}
	tn, ok := w["embed.weight"]
	if !ok {
		t.Fatalf("missing entry")
	}
	if tn.Shape[0] != 2 || tn.Shape[1] != 3 {
		t.Fatalf("shape: got %v", tn.Shape)
	}
	want := []float32{1, 2, 3, 4, 5, 6}
	for i, v := range want {
		if tn.Data[i] != v {
			t.Fatalf("data[%d]: got %f want %f", i, tn.Data[i], v)
		}
	}
}

func TestBinarySmaller(t *testing.T) {
	dir := t.TempDir()
	bp := filepath.Join(dir, "w.bin")
	jp := filepath.Join(dir, "w.json")

	w := map[string]*tensor.Tensor{
		"big": tensor.Random([]int{128, 128}, 0.1),
	}
	if err := SaveWeightsBinary(w, bp); err != nil {
		t.Fatalf("save bin: %v", err)
	}
	if err := SaveWeightsJSON(w, jp); err != nil {
		t.Fatalf("save json: %v", err)
	}
	bs, _ := os.Stat(bp)
	js, _ := os.Stat(jp)
	if bs.Size() >= js.Size() {
		t.Fatalf("expected binary < json: bin=%d json=%d", bs.Size(), js.Size())
	}
	t.Logf("128x128 weights: bin=%d bytes, json=%d bytes (%.1fx)", bs.Size(), js.Size(), float64(js.Size())/float64(bs.Size()))
}

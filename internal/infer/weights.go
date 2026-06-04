package infer

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/justgabe/Nova-U/pkg/tensor"
)

// --- JSON format (interop with train_colab.py) ---

func SaveWeights(weights map[string]*tensor.Tensor, path string) error {
	if strings.HasSuffix(path, ".bin") {
		return SaveWeightsBinary(weights, path)
	}
	return SaveWeightsJSON(weights, path)
}

func LoadWeights(path string) (map[string]*tensor.Tensor, error) {
	if strings.HasSuffix(path, ".bin") {
		return LoadWeightsBinary(path)
	}
	return LoadWeightsJSON(path)
}

func SaveWeightsJSON(weights map[string]*tensor.Tensor, path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Errorf("mkdir: %w", err)
	}
	data, err := json.MarshalIndent(weights, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("write: %w", err)
	}
	return nil
}

func LoadWeightsJSON(path string) (map[string]*tensor.Tensor, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read: %w", err)
	}
	weights := make(map[string]*tensor.Tensor)
	if err := json.Unmarshal(data, &weights); err != nil {
		return nil, fmt.Errorf("unmarshal: %w", err)
	}
	return weights, nil
}

// --- Binary format ---
//
// magic    [4]byte = "NOVA"
// version  uint32  = 1
// nEntries uint32
// per entry:
//   nameLen uint16
//   name    [nameLen]byte
//   nDims   uint8
//   shape   [nDims]int32
//   data    [prod(shape)]float32  little-endian
//
// All multi-byte integers are little-endian.

const (
	binMagic   = "NOVA"
	binVersion = uint32(1)
)

func SaveWeightsBinary(weights map[string]*tensor.Tensor, path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Errorf("mkdir: %w", err)
	}
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create: %w", err)
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	defer w.Flush()

	if _, err := w.WriteString(binMagic); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, binVersion); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, uint32(len(weights))); err != nil {
		return err
	}

	var u32 [4]byte
	for name, t := range weights {
		nb := []byte(name)
		if len(nb) > 0xffff {
			return fmt.Errorf("name too long: %s", name)
		}
		if err := binary.Write(w, binary.LittleEndian, uint16(len(nb))); err != nil {
			return err
		}
		if _, err := w.Write(nb); err != nil {
			return err
		}
		if len(t.Shape) > 255 {
			return fmt.Errorf("too many dims for %s", name)
		}
		if err := w.WriteByte(byte(len(t.Shape))); err != nil {
			return err
		}
		for _, d := range t.Shape {
			if err := binary.Write(w, binary.LittleEndian, int32(d)); err != nil {
				return err
			}
		}
		// raw float32 little-endian; on common targets we could unsafe-cast, but
		// keep it portable.
		for _, v := range t.Data {
			binary.LittleEndian.PutUint32(u32[:], math.Float32bits(v))
			if _, err := w.Write(u32[:]); err != nil {
				return err
			}
		}
	}
	return nil
}

func LoadWeightsBinary(path string) (map[string]*tensor.Tensor, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read: %w", err)
	}
	r := &byteReader{buf: data}

	magic, err := r.readN(4)
	if err != nil || string(magic) != binMagic {
		return nil, fmt.Errorf("bad magic: %q", magic)
	}
	version, err := r.readU32()
	if err != nil {
		return nil, err
	}
	if version != binVersion {
		return nil, fmt.Errorf("unsupported version: %d", version)
	}
	nEntries, err := r.readU32()
	if err != nil {
		return nil, err
	}

	weights := make(map[string]*tensor.Tensor, nEntries)
	for i := uint32(0); i < nEntries; i++ {
		nameLen, err := r.readU16()
		if err != nil {
			return nil, err
		}
		nameBytes, err := r.readN(int(nameLen))
		if err != nil {
			return nil, err
		}
		name := string(nameBytes)
		nDims, err := r.readByte()
		if err != nil {
			return nil, err
		}
		shape := make([]int, nDims)
		total := 1
		for d := 0; d < int(nDims); d++ {
			v, err := r.readI32()
			if err != nil {
				return nil, err
			}
			shape[d] = int(v)
			total *= int(v)
		}
		raw, err := r.readN(total * 4)
		if err != nil {
			return nil, err
		}
		fdata := make([]float32, total)
		for j := 0; j < total; j++ {
			fdata[j] = math.Float32frombits(binary.LittleEndian.Uint32(raw[j*4:]))
		}
		weights[name] = tensor.New(fdata, shape)
	}
	return weights, nil
}

type byteReader struct {
	buf []byte
	pos int
}

func (r *byteReader) readN(n int) ([]byte, error) {
	if r.pos+n > len(r.buf) {
		return nil, io.ErrUnexpectedEOF
	}
	b := r.buf[r.pos : r.pos+n]
	r.pos += n
	return b, nil
}

func (r *byteReader) readByte() (byte, error) {
	b, err := r.readN(1)
	if err != nil {
		return 0, err
	}
	return b[0], nil
}

func (r *byteReader) readU16() (uint16, error) {
	b, err := r.readN(2)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint16(b), nil
}

func (r *byteReader) readU32() (uint32, error) {
	b, err := r.readN(4)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(b), nil
}

func (r *byteReader) readI32() (int32, error) {
	v, err := r.readU32()
	return int32(v), err
}

func ParamsFromPyTorch(pytorchWeights map[string][]float32, nameMap map[string]string) map[string]*tensor.Tensor {
	goWeights := make(map[string]*tensor.Tensor)
	for pyName, goName := range nameMap {
		if data, ok := pytorchWeights[pyName]; ok {
			goWeights[goName] = tensor.New(data, []int{len(data)})
		}
	}
	return goWeights
}

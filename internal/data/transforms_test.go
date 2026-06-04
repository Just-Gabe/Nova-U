package data

import (
	"path/filepath"
	"testing"
)

func TestSimpleTokenizerRoundtrip(t *testing.T) {
	tk := NewSimpleTokenizer([]string{"hello, world! ção"})

	text := "hello, ção"
	ids := tk.Encode(text)
	out := tk.Decode(ids)
	if out != text {
		t.Fatalf("roundtrip: got %q want %q", out, text)
	}
}

func TestSimpleTokenizerDeterministic(t *testing.T) {
	tk1 := NewSimpleTokenizer([]string{"abc def"})
	tk2 := NewSimpleTokenizer([]string{"def abc"})
	if tk1.CharToID['a'] != tk2.CharToID['a'] {
		t.Fatalf("non-deterministic vocab ordering")
	}
}

func TestSimpleTokenizerUnknownToPad(t *testing.T) {
	tk := NewSimpleTokenizer([]string{"abc"})
	ids := tk.Encode("axc")
	if ids[1] != 0 {
		t.Fatalf("unknown char should map to PAD(0), got %d", ids[1])
	}
}

func TestSimpleTokenizerVocabSizeIncludesReserved(t *testing.T) {
	tk := NewSimpleTokenizer([]string{"abc"})
	if tk.VocabSize() != 6 { // 3 reserved + 3 chars
		t.Fatalf("vocab size: got %d want 6", tk.VocabSize())
	}
}

func TestSimpleTokenizerPersistence(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "vocab.json")

	original := NewSimpleTokenizer([]string{"hello world ção"})
	if err := original.Save(path); err != nil {
		t.Fatalf("save: %v", err)
	}
	loaded, err := LoadSimpleTokenizer(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	text := "ção world"
	if loaded.Decode(loaded.Encode(text)) != original.Decode(original.Encode(text)) {
		t.Fatalf("loaded tokenizer behaves differently")
	}
}

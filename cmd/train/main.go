package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/justgabe/Nova-U/internal/data"
)

func main() {
	dataPath := flag.String("data", "data/train.txt", "path to training text")
	vocabPath := flag.String("vocab-out", "checkpoints/vocab.json", "path to write the learned vocab")
	flag.Parse()

	log.Println("Nova-U Go-side training is currently a stub.")
	log.Println("Real backpropagation is not yet implemented in Go.")
	log.Println("Use scripts/train_colab.py to train on GPU and export weights:")
	log.Println("  python scripts/train_colab.py --data data/train.txt --export model.bin")
	log.Println("Then load with cmd/generate -weights model.bin -vocab checkpoints/vocab.json")
	log.Println()

	// We can still help by producing a deterministic vocab file from the
	// dataset, which the Python script and the Go inference both rely on.
	raw, err := os.ReadFile(*dataPath)
	if err != nil {
		log.Fatalf("read data: %v", err)
	}
	tok := data.NewSimpleTokenizer([]string{string(raw)})
	if err := os.MkdirAll(filepath.Dir(*vocabPath), 0755); err != nil {
		log.Fatalf("mkdir: %v", err)
	}
	if err := tok.Save(*vocabPath); err != nil {
		log.Fatalf("save vocab: %v", err)
	}
	log.Printf("vocab written to %s (size=%d)", *vocabPath, tok.VocabSize())
}

func init() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Nova-U train (vocab-only stub)\n\n")
		fmt.Fprintf(os.Stderr, "Usage: train [flags]\n\n")
		fmt.Fprintf(os.Stderr, "Flags:\n")
		flag.PrintDefaults()
	}
}

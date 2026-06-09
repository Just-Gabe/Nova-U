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
	dataPath := flag.String("data", "", "path to training text (.txt)")
	jsonlPath := flag.String("jsonl", "", "path to JSONL dataset with conversations")
	vocabPath := flag.String("vocab-out", "checkpoints/vocab.json", "path to write the learned vocab")
	flag.Parse()

	if *dataPath == "" && *jsonlPath == "" {
		log.Fatal("either -data (txt) or -jsonl must be provided")
	}

	log.Println("Nova-U Go-side training is currently a stub.")
	log.Println("Real backpropagation is not yet implemented in Go.")
	log.Println("Use scripts/train_colab.py to train on GPU and export weights:")
	if *jsonlPath != "" {
		log.Printf("  python scripts/train_colab.py --jsonl %s --export model.bin", *jsonlPath)
	}
	log.Printf("  python scripts/train_colab.py --data data/train.txt --export model.bin")
	log.Println("Then load with cmd/generate -weights model.bin -vocab checkpoints/vocab.json")
	log.Println()

	texts, err := loadTexts(*dataPath, *jsonlPath)
	if err != nil {
		log.Fatalf("load data: %v", err)
	}

	tok := data.NewSimpleTokenizer(texts)
	if err := os.MkdirAll(filepath.Dir(*vocabPath), 0755); err != nil {
		log.Fatalf("mkdir: %v", err)
	}
	if err := tok.Save(*vocabPath); err != nil {
		log.Fatalf("save vocab: %v", err)
	}
	log.Printf("vocab written to %s (size=%d)", *vocabPath, tok.VocabSize())
}

func loadTexts(dataPath, jsonlPath string) ([]string, error) {
	if jsonlPath != "" {
		return data.ReadJSONLDataset(jsonlPath)
	}
	raw, err := os.ReadFile(dataPath)
	if err != nil {
		return nil, err
	}
	return []string{string(raw)}, nil
}

func init() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Nova-U train (vocab-only stub)\n\n")
		fmt.Fprintf(os.Stderr, "Usage: train [flags]\n\n")
		fmt.Fprintf(os.Stderr, "Flags:\n")
		flag.PrintDefaults()
	}
}

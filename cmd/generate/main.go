package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/justgabe/Nova-U/internal/data"
	"github.com/justgabe/Nova-U/internal/infer"
	"github.com/justgabe/Nova-U/internal/model"
)

func main() {
	weightsPath := flag.String("weights", "", "path to model weights (.bin or .json)")
	vocabPath := flag.String("vocab", "", "path to vocab.json from training")
	prompt := flag.String("prompt", "", "input prompt")
	maxLen := flag.Int("max-len", 128, "maximum generation length")
	temp := flag.Float64("temp", 0.8, "sampling temperature")
	topK := flag.Int("topk", 40, "top-k sampling parameter")
	dModel := flag.Int("dmodel", 128, "model dimension")
	seqLen := flag.Int("seq-len", 128, "max sequence length")
	vocabFlag := flag.Int("vocab-size", 0, "vocab size (ignored if -vocab is given)")
	flag.Parse()

	var tok *data.SimpleTokenizer
	vocabSize := *vocabFlag
	if *vocabPath != "" {
		var err error
		tok, err = data.LoadSimpleTokenizer(*vocabPath)
		if err != nil {
			log.Fatalf("load vocab: %v", err)
		}
		vocabSize = tok.VocabSize()
	} else if vocabSize == 0 {
		vocabSize = 256
		log.Printf("no -vocab given; defaulting -vocab-size=%d (codepoint mode)", vocabSize)
	}

	cfg := model.DefaultARConfig()
	cfg.UNetConfig.VocabSize = vocabSize
	cfg.UNetConfig.DModel = *dModel
	cfg.UNetConfig.MaxSeqLen = *seqLen
	cfg.MaxGenLen = *maxLen
	cfg.Temperature = float32(*temp)
	cfg.TopK = *topK

	log.Printf("Nova-U generation configuration:")
	log.Printf("  vocab_size=%d, d_model=%d", vocabSize, *dModel)
	log.Printf("  max_len=%d, temp=%.2f, top_k=%d", *maxLen, *temp, *topK)

	ar := model.NewARModel(cfg)

	if *weightsPath != "" {
		log.Printf("loading weights from %s...", *weightsPath)
		weights, err := infer.LoadWeights(*weightsPath)
		if err != nil {
			log.Fatalf("failed to load weights: %v", err)
		}
		ar.SetParams(weights)
		log.Printf("loaded %d parameter tensors", len(weights))
	}

	var context []int
	if *prompt != "" {
		if tok != nil {
			context = tok.Encode(*prompt)
		} else {
			for _, ch := range *prompt {
				context = append(context, int(ch))
			}
		}
		log.Printf("prompt: %q (%d tokens)", *prompt, len(context))
	}

	log.Println("generating...")
	output := ar.Generate(context, *maxLen)

	var result string
	if tok != nil {
		result = tok.Decode(output)
	} else {
		for _, id := range output {
			if id < 32 {
				continue
			}
			result += string(rune(id))
		}
	}

	fmt.Println("\nGenerated output:")
	fmt.Println("---------------")
	fmt.Println(result)
	fmt.Println("---------------")
}

func init() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Nova-U Generation\n\n")
		fmt.Fprintf(os.Stderr, "Usage: generate [flags]\n\n")
		fmt.Fprintf(os.Stderr, "Flags:\n")
		flag.PrintDefaults()
	}
}

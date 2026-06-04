package train

import "github.com/justgabe/Nova-U/internal/model"

type Config struct {
	Model         model.ARConfig
	BatchSize     int
	Epochs        int
	LearningRate  float32
	WeightDecay   float32
	DataPath      string
	CheckpointDir string
	LogInterval   int
	SaveInterval  int
	MaxSteps      int
	NumWorkers    int
}

func DefaultConfig() Config {
	return Config{
		Model:         model.DefaultARConfig(),
		BatchSize:     16,
		Epochs:        10,
		LearningRate:  3e-4,
		WeightDecay:   1e-4,
		DataPath:      "data/train.txt",
		CheckpointDir: "checkpoints",
		LogInterval:   10,
		SaveInterval:  1000,
		MaxSteps:      100000,
		NumWorkers:    4,
	}
}

package data

import (
	"bufio"
	"encoding/json"
	"os"
)

type Conversation struct {
	Conversations []Message `json:"conversations"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

const (
	SystemMarker    = "<|sys|>"
	UserMarker      = "<|usr|>"
	AssistantMarker = "<|ast|>"
	EndMarker       = "<|end|>"
)

func FormatConversation(conv *Conversation) string {
	var b []byte
	for _, msg := range conv.Conversations {
		switch msg.Role {
		case "system":
			b = append(b, SystemMarker...)
			b = append(b, '\n')
			b = append(b, msg.Content...)
			b = append(b, '\n')
			b = append(b, EndMarker...)
			b = append(b, '\n')
		case "user":
			b = append(b, UserMarker...)
			b = append(b, '\n')
			b = append(b, msg.Content...)
			b = append(b, '\n')
			b = append(b, EndMarker...)
			b = append(b, '\n')
		case "assistant":
			b = append(b, AssistantMarker...)
			b = append(b, '\n')
			b = append(b, msg.Content...)
			b = append(b, '\n')
			b = append(b, EndMarker...)
			b = append(b, '\n')
		}
	}
	return string(b)
}

func ReadJSONLDataset(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var texts []string
	scanner := bufio.NewScanner(f)
	buf := make([]byte, 0, 1024*1024)
	scanner.Buffer(buf, 10*1024*1024)

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		var conv Conversation
		if err := json.Unmarshal(line, &conv); err != nil {
			return nil, err
		}
		texts = append(texts, FormatConversation(&conv))
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return texts, nil
}

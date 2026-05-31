package main

import (
	"encoding/json"
	"os"
)

func writeJSON(path string, v interface{}) {
	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		panic(err)
	}
}

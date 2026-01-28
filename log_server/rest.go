package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
)

func get_playgrounds(w http.ResponseWriter, r *http.Request) {
	if r.Method != "GET" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	playground := "/app/playground"
	if path, ok := os.LookupEnv("MAPPED_PLAYGROUND_PATH"); ok {
		playground = path
	}

	files, err := os.ReadDir(playground)
	if err != nil {
		fmt.Fprintf(os.Stderr, "GET /api/playgrounds : %v\n", err)
		http.Error(w, "Technical error", http.StatusInternalServerError)
		return
	}

	var folders []string
	for _, file := range files {
		if file.IsDir() {
			folders = append(folders, file.Name())
		}
	}

	w.Header().Set("Content-type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	json.NewEncoder(w).Encode(map[string][]string{
		"folders": folders,
	})
}

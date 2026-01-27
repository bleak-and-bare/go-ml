package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/bleak-and-bare/go-ml/log_server/command"
	"github.com/bleak-and-bare/go-ml/log_server/ws"
)

func main() {
	port := "8000"
	if p, ok := os.LookupEnv("SERVER_PORT"); ok {
		port = p
	}

	hub := ws.NewHub()
	go hub.Run()

	c := command.NewCmdController()
	go c.Run()

	http.HandleFunc("/api/playgrounds", get_playgrounds)

	http.HandleFunc("/ws", func(w http.ResponseWriter, r *http.Request) {
		ServeWS(hub, c.Commands(), w, r)
	})

	fmt.Printf("Listening on port %v\n", port)
	err := http.ListenAndServe(":"+port, nil)
	if err != nil {
		log.Fatal("http.ListenAndServe : ", err)
	}
}

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
	json.NewEncoder(w).Encode(map[string][]string{
		"folders": folders,
	})
}

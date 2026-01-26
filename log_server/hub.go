package main

import "fmt"

type Hub struct {
	// registered clients
	clients map[*Client]bool

	// message from clients
	broadcast chan []byte

	register   chan *Client
	unregister chan *Client
}

func NewHub() *Hub {
	return &Hub{
		clients:    make(map[*Client]bool),
		broadcast:  make(chan []byte),
		register:   make(chan *Client),
		unregister: make(chan *Client),
	}
}

func (h *Hub) Register(c *Client) { h.register <- c }

func (h *Hub) Unregister(c *Client) { h.unregister <- c }

func (h *Hub) Broadcast(msg []byte) { h.broadcast <- msg }

func (h *Hub) Run() {
	for {
		select {
		case client := <-h.register:
			fmt.Printf("Hub.Run : new client %p\n", client)
			h.clients[client] = true
		case client := <-h.unregister:
			if _, ok := h.clients[client]; ok {
				fmt.Printf("Hub.Run : %p unregistered\n", client)
				client.Close()
				delete(h.clients, client)
			}
		case message := <-h.broadcast:
			for client := range h.clients {
				select {
				case client.Send() <- message:
				default:
					client.Close()
					delete(h.clients, client)
				}
			}
		}
	}
}

package ws

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"sync/atomic"
	"time"

	"github.com/bleak-and-bare/go-ml/message"
	"github.com/gorilla/websocket"
)

type Client struct {
	hub           *Hub
	conn          *websocket.Conn
	cmd           *exec.Cmd
	send          chan []byte
	ctx           context.Context
	cancel        func()
	PausedProcess atomic.Bool
}

// create then register a new client
func NewClient(hub *Hub, conn *websocket.Conn) *Client {
	ctx, cancel := context.WithCancel(context.Background())
	client := &Client{
		hub:           hub,
		conn:          conn,
		cmd:           nil,
		send:          make(chan []byte, 256),
		PausedProcess: atomic.Bool{},
	}

	client.ctx = ctx
	client.cancel = cancel

	hub.Register(client)
	return client
}

func (c *Client) Send() chan<- []byte {
	return c.send
}

func (c *Client) SetExecCmd(cmd *exec.Cmd) {
	c.cmd = cmd
}

func (c *Client) GetExecCmd() *exec.Cmd {
	return c.cmd
}

func (c *Client) Context() context.Context {
	return c.ctx
}

func (c *Client) Unregister(cmd chan<- Command) {
	c.cancel()
	if c.cmd != nil {
		cmd <- Command{
			Client:  c,
			Message: message.NewAbort(),
		}
	}

	c.hub.Unregister(c)
	c.conn.Close()
}

// pumps message and push to the hub
func (c *Client) ReadPump(cmd chan<- Command) {
	defer c.Unregister(cmd)

	c.conn.SetReadLimit(MAX_MSG_SIZE)
	c.conn.SetReadDeadline(time.Now().Add(PONG_WAIT))
	c.conn.SetPongHandler(func(string) error { // extend the deadline
		c.conn.SetReadDeadline(time.Now().Add(PONG_WAIT))
		return nil
	})

	for {
		_, msg, err := c.conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseAbnormalClosure) {
				fmt.Fprintf(os.Stderr, "Client.ReadPump : %v\n", err)
			}

			return
		}
		// msg = bytes.TrimSpace(bytes.ReplaceAll(msg, new_line, space))

		var m message.Message
		if err := json.Unmarshal(msg, &m); err != nil {
			err_bytes, _ := json.Marshal(message.NewError(err.Error(), false))

			select {
			case <-c.ctx.Done():
			case c.send <- err_bytes:
			}

			continue
		}

		cmd <- Command{
			Client:  c,
			Message: m,
		}
	}
}

// pumps message from the hub to connection
func (c *Client) WritePump() {
	ticker := time.NewTicker(PING_PERIOD)
	defer func() {
		ticker.Stop()
		c.conn.Close()
	}()

	for {
		select {
		case msg, ok := <-c.send:
			c.conn.SetWriteDeadline(time.Now().Add(WRITE_WAIT))
			if !ok {
				c.conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}

			w, err := c.conn.NextWriter(websocket.TextMessage)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Client.WritePump : failed to get writer %v\n", err)
				return
			}

			w.Write(msg)

			// msg_drain:
			// 	for {
			// 		select {
			// 		case <-c.ctx.Done():
			// 		case msg, ok := <-c.send:
			// 			if !ok {
			// 				break msg_drain
			// 			}
			//
			// 			w.Write(msg)
			//
			// 		default:
			// 			break msg_drain
			// 		}
			// 	}

			if err := w.Close(); err != nil {
				fmt.Fprintf(os.Stderr, "Client.WritePump : failed to close writer %v\n", err)
				return
			}

		case <-ticker.C:
			c.conn.SetWriteDeadline(time.Now().Add(WRITE_WAIT))
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				fmt.Fprintf(os.Stderr, "Client.WritePump : failed to write ping message %v\n", err)
				return
			}
		}
	}
}

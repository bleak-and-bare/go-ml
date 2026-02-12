package main

import (
	"encoding/json"
	"time"

	"github.com/bleak-and-bare/go-ml/log_server/stat"
	"github.com/bleak-and-bare/go-ml/log_server/ws"
	"github.com/bleak-and-bare/go-ml/message"
)

func monitor_processes(c *ws.Client, dt time.Duration) {
	cmd := c.GetExecCmd()
	if cmd == nil || cmd.Process == nil {
		return
	}

	cpu, err := stat.ReadCPUStat(cmd.Process.Pid)
	if err == nil {
		mem, mem_err := stat.ReadMemStat(cmd.Process.Pid)
		err = mem_err

		if err == nil {
			msg_bytes, _ := json.Marshal(message.Message{
				Type: message.STATS,
				Data: stat.NewStat(cpu, mem, dt),
			})

			select {
			case <-c.Context().Done():
			case c.Send() <- msg_bytes:
			}
		}
	}

	if err != nil {
		err_bytes, _ := json.Marshal(message.Message{
			Type: message.ERROR,
			Data: err.Error(),
		})

		select {
		case <-c.Context().Done():
		case c.Send() <- err_bytes:
		}
	}
}

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
			stats := stat.NewStat(cpu, mem, dt)
			msg_bytes, _ := json.Marshal(message.NewStats(message.StatStruct{
				Process: message.ProcessInfos{
					User:   int64(stats.Process.User),
					System: int64(stats.Process.System),
				},
				Global:  int64(stats.Global),
				RSS:     stats.RSS,
				PeakRSS: stats.PeakRSS,
				Delta:   int64(stats.Delta),
			}))

			select {
			case <-c.Context().Done():
			case c.Send() <- msg_bytes:
			}
		}
	}

	if err != nil {
		err_bytes, _ := json.Marshal(message.NewError(err.Error(), false))

		select {
		case <-c.Context().Done():
		case c.Send() <- err_bytes:
		}
	}
}

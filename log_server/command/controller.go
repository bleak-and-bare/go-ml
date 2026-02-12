package command

import (
	"fmt"
	"os"
	"syscall"
	"time"

	"github.com/bleak-and-bare/go-ml/log_server/ws"
	"github.com/bleak-and-bare/go-ml/message"
)

type CmdController struct {
	cmds chan ws.Command
}

func NewCmdController() *CmdController {
	return &CmdController{make(chan ws.Command, 8)}
}

func (c *CmdController) Commands() chan<- ws.Command {
	return c.cmds
}

func (c *CmdController) Run() {
	for cmd := range c.cmds {
		switch cmd.Type {
		case message.EXECUTE:
			if cmd.Client.GetExecCmd() != nil {
				send_error_to_client(cmd.Client, fmt.Errorf("You can only run one process"))
				continue
			}
			fmt.Printf("CmdController.Run : %p running process\n", cmd.Client)

			data, ok := cmd.Data.(string)
			if !ok {
				send_error_to_client(cmd.Client, fmt.Errorf("message.data should be a folder name"))
				continue
			}

			exec_cmd, err := CreateGoRunCmd(data)
			if err != nil {
				fmt.Fprintf(os.Stderr, "CmdController.Run : %v\n", err)
				send_error_to_client(cmd.Client, err)
				continue
			}

			if err != stream_output(exec_cmd, cmd.Client) {
				fmt.Fprintf(os.Stderr, "CmdController.Run : %v\n", err)
				continue
			}

			cmd.Client.SetExecCmd(exec_cmd)
			start := time.Now()
			err = exec_cmd.Start()

			if err != nil {
				fmt.Fprintf(os.Stderr, "CmdController.Run : %v\n", err)
				send_error_to_client(cmd.Client, err)
				continue
			}

			go func() {
				fmt.Printf("spawned PID=%d\n", exec_cmd.Process.Pid)
				exec_cmd.Wait()

				// for consistency
				cmd.Client.PausedProcess.Store(false)

				fmt.Printf("CmdController.Run : %p process finished running\n", cmd.Client)
				fmt.Println(exec_cmd.ProcessState)

				notify_exec_finished_to_client(start, cmd.Client)
				cmd.Client.SetExecCmd(nil)
			}()

			notify_req_fulfilled_to_client(cmd.Type, cmd.Client)

		case message.ABORT:
			exec_cmd := cmd.Client.GetExecCmd()
			if exec_cmd == nil {
				fmt.Fprintf(os.Stderr, "CmdController.Run : %p didn't spawn any process\n", cmd.Client)
				continue
			}
			fmt.Printf("CmdController.Run : %p terminating process\n", cmd.Client)

			go func() {
				err := fmt.Errorf("")
				if !cmd.Client.PausedProcess.Load() {
					err = syscall.Kill(-exec_cmd.Process.Pid, syscall.SIGTERM)
				}

				if err != nil {
					err = syscall.Kill(-exec_cmd.Process.Pid, syscall.SIGKILL)
					if err != nil {
						send_error_to_client(cmd.Client, err)
						return
					}
				}

				send_info_to_client(cmd.Client, "Process terminated.")
			}()

			notify_req_fulfilled_to_client(cmd.Type, cmd.Client)

		case message.PAUSE:
			exec_cmd := cmd.Client.GetExecCmd()
			if exec_cmd == nil {
				fmt.Fprintf(os.Stderr, "CmdController.Run : %p didn't spawn any process\n", cmd.Client)
				continue
			}
			fmt.Printf("CmdController.Run : %p pausing process\n", cmd.Client)

			go func() {
				if err := syscall.Kill(-exec_cmd.Process.Pid, syscall.SIGSTOP); err != nil {
					send_error_to_client(cmd.Client, err)
				} else {
					cmd.Client.PausedProcess.Store(true)
				}
			}()

			notify_req_fulfilled_to_client(cmd.Type, cmd.Client)

		case message.RESUME:
			exec_cmd := cmd.Client.GetExecCmd()
			if exec_cmd == nil {
				fmt.Fprintf(os.Stderr, "CmdController.Run : %p didn't spawn any process\n", cmd.Client)
				continue
			}
			fmt.Printf("CmdController.Run : %p resuming process\n", cmd.Client)

			go func() {
				if err := syscall.Kill(-exec_cmd.Process.Pid, syscall.SIGCONT); err != nil {
					send_error_to_client(cmd.Client, err)
				} else {
					cmd.Client.PausedProcess.Store(false)
				}
			}()

			notify_req_fulfilled_to_client(cmd.Type, cmd.Client)
		}
	}
}

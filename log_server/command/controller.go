package command

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"slices"
	"syscall"

	"github.com/bleak-and-bare/go-ml/log_server/ws"
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

func send_info_to_client(client *ws.Client, msg string) {
	msg_bytes, _ := json.Marshal(ws.Message{
		Type: ws.INFO,
		Data: msg,
	})
	client.Send() <- msg_bytes
}

func send_error_to_client(client *ws.Client, err error) {
	err_bytes, _ := json.Marshal(ws.Message{
		Type: ws.ERROR,
		Data: err.Error(),
	})
	client.Send() <- err_bytes
}

func stream_frame(r io.Reader, f func([]byte)) {
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := slices.Clone(scanner.Bytes())
		f(line)
	}
}

func stream_output(cmd *exec.Cmd, client *ws.Client) error {
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return err
	}

	stdio, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}

	go stream_frame(stdio, func(b []byte) { client.Send() <- b })
	go stream_frame(stderr, func(b []byte) { client.Send() <- b })

	return nil
}

func (c *CmdController) Run() {
	for cmd := range c.cmds {
		switch cmd.Type {
		case ws.EXECUTE:
			if cmd.Client.GetExecCmd() != nil {
				send_error_to_client(cmd.Client, fmt.Errorf("You can only run one process"))
				continue
			}

			data, ok := cmd.Data.(string)
			if !ok {
				send_error_to_client(cmd.Client, fmt.Errorf("message.data should be a folder name"))
			}

			exec_cmd, err := CreateCmd(data)
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
			err = exec_cmd.Start()

			if err != nil {
				fmt.Fprintf(os.Stderr, "CmdController.Run : %v\n", err)
				send_error_to_client(cmd.Client, err)
				continue
			}

			go func() {
				exec_cmd.Wait()
				cmd.Client.SetExecCmd(nil)
			}()

		case ws.ABORT:
			exec_cmd := cmd.Client.GetExecCmd()
			if exec_cmd == nil {
				fmt.Fprintf(os.Stderr, "CmdController.Run : %p didn't spawn any process\n", cmd.Client)
				continue
			}

			go func() {
				err := syscall.Kill(-exec_cmd.Process.Pid, syscall.SIGTERM)
				if err != nil {
					err = syscall.Kill(-exec_cmd.Process.Pid, syscall.SIGKILL)
					if err != nil {
						send_error_to_client(cmd.Client, err)
					}
					return
				}

				send_info_to_client(cmd.Client, "Process terminated.")
			}()

		case ws.PAUSE:
			fmt.Printf("CmdController.Run : %v pausing process\n", cmd.Client)
			exec_cmd := cmd.Client.GetExecCmd()
			if exec_cmd == nil {
				fmt.Fprintf(os.Stderr, "CmdController.Run : %p didn't spawn any process\n", cmd.Client)
				continue
			}

			go func() {
				if err := syscall.Kill(-exec_cmd.Process.Pid, syscall.SIGSTOP); err != nil {
					send_error_to_client(cmd.Client, err)
				}
			}()

		case ws.RESUME:
			fmt.Printf("CmdController.Run : %v resuming process\n", cmd.Client)
			exec_cmd := cmd.Client.GetExecCmd()
			if exec_cmd == nil {
				fmt.Fprintf(os.Stderr, "CmdController.Run : %p didn't spawn any process\n", cmd.Client)
				continue
			}

			go func() {
				if err := syscall.Kill(-exec_cmd.Process.Pid, syscall.SIGCONT); err != nil {
					send_error_to_client(cmd.Client, err)
				}
			}()
		}
	}
}

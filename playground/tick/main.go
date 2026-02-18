package main

import (
	"fmt"
	"time"
)

func main() {
	for i := range 200 {
		fmt.Println(i)
		time.Sleep(time.Second)
	}
}

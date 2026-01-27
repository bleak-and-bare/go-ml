package main

import (
	"fmt"
	"time"
)

func main() {
	var i int
	for {
		fmt.Println(i)
		i++
		time.Sleep(2 * time.Second)
	}
}

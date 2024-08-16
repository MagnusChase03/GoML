package main

import (
    "fmt"

    "github.com/MagnusChase03/GoML/ml"
)

func main() {
    nn, err := ml.NewFFNN([]int{10, 6, 3})
    if err != nil {
        fmt.Printf("%w\n", err)
        return
    }

    fmt.Printf("%v\n", nn.Weights[0].Data)
}

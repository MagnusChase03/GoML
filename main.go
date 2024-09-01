package main

import (
	"fmt"
	"os"

	"github.com/MagnusChase03/GoML/ml"
)

func main() { 
    n, err := ml.NewFFNN([]int{3, 2, 1});
    if err != nil {
        fmt.Fprintf(os.Stderr, "%v", err);
        return;
    }

    for i := 0; i < 100; i++ {
        inputs := [][]float64{{1.0, 2.0, 3.0}, {3.0, 2.0, 1.0}};
        outputs, err := n.Forward(inputs);
        if err != nil {
            fmt.Fprintf(os.Stderr, "%v", err);
            return;
        }
        fmt.Printf("%v\n", outputs);

        err = n.Backward(inputs, [][]float64{{1.0}, {3.0}}, outputs, 0.01);
        if err != nil {
            fmt.Fprintf(os.Stderr, "%v", err);
            return;
        }
        n.Flush();
    }
}

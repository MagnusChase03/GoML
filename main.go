package main

import (
    "fmt"
    "os"

    "github.com/MagnusChase03/GoML/ml"
)

func main() { 
    d, err := ml.NewDenseLayer(3, 2);
    if err != nil {
        fmt.Fprintf(os.Stderr, "%v\n", err);
    }

    for i := 0; i < 10; i++ {
        inputs := [][]float64{{1.0, 2.0, 3.0}, {3.0, 2.0, 1.0}};
        outputs, err := d.Forward(inputs);
        if err != nil {
            fmt.Fprintf(os.Stderr, "%v\n", err);
        }
        fmt.Printf("%v\n", outputs);

        deltas := [][]float64{{outputs[0][0] - 1.0, outputs[0][1]}, {outputs[1][0] - 1.0, outputs[1][1]}};
        _, err = d.Backward(inputs, deltas, 0.001);
        if err != nil {
            fmt.Fprintf(os.Stderr, "%v\n", err);
        }
        d.Flush();
    }
}

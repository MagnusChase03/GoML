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

    inputs := [][]float64{
        {0, 0, 0}, 
        {0, 1, 0},
        {0, 1, 1},
        {1, 0, 0},
        {1, 0, 1},
        {1, 1, 1},
    };
    targets := [][]float64{
        {0}, 
        {1},
        {0},
        {1},
        {0},
        {1},
    };
    n.Train(inputs, targets, 0.01, 100, 2);

    outputs, err := n.Forward([][]float64{{1, 1, 0}, {0, 0, 1}});
    if err != nil {
        fmt.Fprintf(os.Stderr, "%v", err);
        return;
    }
    fmt.Printf("%v\n", outputs);

    n.Save("./model.json")
    
   /*n2, err := ml.LoadFFNN("./model.json");
    if err != nil {
        fmt.Fprintf(os.Stderr, "%v", err);
        return;
    }
    fmt.Printf("%v\n", n.Layers[0].Weights);
    fmt.Printf("%v\n", n2.Layers[0].Weights);*/
}

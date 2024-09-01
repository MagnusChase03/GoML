package ml

import (
    "fmt"
    "math/rand"
    "sync"
)

type DenseLayer struct {
    Shape []int

    Weights [][]float64
    Bias []float64

    Delta [][]float64
    DeltaBias []float64
}

/*
Creates a new dense layer.

Arguments:
    - i (int): The number of nodes in the input layer.
    - o (int): The number of nodes in the output layer.

Returns:
    - *DenseLayer: The dense layer with given dimensions.
    - error: An error if any occured.

Example:
    d, err := NewDenseLayer(3, 2);
*/
func NewDenseLayer(i int, o int) (*DenseLayer, error) {
    if (i < 1 || o < 1) {
        return nil, fmt.Errorf("Error: Invalid dense layer dimensions.");
    }

    var wg sync.WaitGroup;
    wg.Add(o);

    weights := make([][]float64, o);
    delta := make([][]float64, o);
    for r := 0; r < o; r++ {
        weights[r] = make([]float64, i);
        delta[r] = make([]float64, i);
        go func(row int) {
            defer wg.Done();
            for c := 0; c < i; c++ {
                weights[r][c] = rand.Float64() / 10.0;
            }
        }(r);
    }

    wg.Wait();
    return &DenseLayer{
        Shape: []int{i, o},
        Weights: weights,
        Bias: make([]float64, o),
        Delta: delta,
        DeltaBias: make([]float64, o),
    }, nil;
}

/*
Does a batch forward pass through the dense layer.

Arguments
    - inputs ([][]float64): The set of inputs to be forwarded.

Returns:
    - [][]float64: The output of each batch.
    - error: The error that occured if any.

Example:
    y, err := d.Forward(inputBatch);
*/
func (d *DenseLayer) Forward(inputs [][]float64) ([][]float64, error) {
    if len(inputs) < 1 || len(inputs[0]) != d.Shape[0] {
        return nil, fmt.Errorf("Error: Invalid input dimensions.");
    }

    result := make([][]float64, len(inputs));
    var wg sync.WaitGroup;
    wg.Add(len(inputs));
    for i := 0; i < len(inputs); i++ {
        result[i] = make([]float64, d.Shape[1]);
        go func(set int) {
            defer wg.Done();
            partial := make([]float64, d.Shape[1]);

            var wg2 sync.WaitGroup;
            wg2.Add(len(d.Weights));
            for j := 0; j < d.Shape[1]; j++ {
                go func(row int) {
                    defer wg2.Done();
                    for k := 0; k < d.Shape[0]; k++ {
                        partial[row] += d.Weights[row][k] * inputs[set][k];
                    }
                    partial[row] += d.Bias[row];
                }(j);
            }
            wg2.Wait();
            result[set] = partial;
        }(i);
    }

    wg.Wait();
    return result, nil;
}

/*
Does a batch backward propagation to determine deltas.

Arguments:
    - inputs ([][]float64): The input batch.
    - deltas ([][])float64: The output error batch.
    - lr (float64): The learning rate.

Returns:
    - [][]float64: The deltas for the previous layer. 
    - error: The error if any occured.

Example:
    de, err := d.Backward(inputs, deltas, 0.001);
*/
func (d *DenseLayer) Backward(
    inputs [][]float64, 
    deltas [][]float64, 
    lr float64,
) ([][]float64, error) {
    if (len(inputs) < 1 || 
    len(inputs) != len(deltas) ||
    len(inputs[0]) != d.Shape[0] ||
    len(deltas[0]) != d.Shape[1]) {
        return nil, fmt.Errorf("Error: Invalid inputs or deltas dimensions.");
    }

    de := make([][]float64, len(inputs));
    mutexes := make([][]sync.Mutex, d.Shape[1]);
    for i := 0; i < d.Shape[1]; i++ {
        mutexes[i] = make([]sync.Mutex, d.Shape[0]);
    }
    biasMutexes := make([]sync.Mutex, d.Shape[1]);

    var wg sync.WaitGroup;
    wg.Add(len(inputs));
    for i := 0; i < len(inputs); i++ {
        de[i] = make([]float64, d.Shape[0]);
        go func(set int) {
            defer wg.Done();

            var wg2 sync.WaitGroup;
            wg2.Add(d.Shape[1]);
            for j := 0; j < d.Shape[1]; j++ {
                go func(row int) {
                    defer wg2.Done();
                    biasMutexes[row].Lock();
                    d.DeltaBias[row] += deltas[set][row] * lr;
                    biasMutexes[row].Unlock();
                    for k := 0; k < d.Shape[0]; k++ {
                        mutexes[row][k].Lock();
                        d.Delta[row][k] += deltas[set][row] * inputs[set][k] * lr;
                        mutexes[row][k].Unlock();

                        de[set][k] += deltas[set][row] * d.Weights[row][k];
                    }
                }(j);
            }
            wg2.Wait();
        }(i);
    }

    wg.Wait();
    return de, nil;
}

/*
Updates weights based on deltas.

Arguments:
    - N/A

Returns:
    - N/A

Example:
    d.Flush();
*/
func (d *DenseLayer) Flush() {
    var wg sync.WaitGroup;
    wg.Add(d.Shape[1])
    for i := 0; i < d.Shape[1]; i++ {
        go func(row int) {
            defer wg.Done();
            d.Bias[row] -= d.DeltaBias[row];
            d.DeltaBias[row] = 0.0;
            for j := 0; j < d.Shape[0]; j++ {
                d.Weights[row][j] -= d.Delta[row][j];
                d.Delta[row][j] = 0.0;
            }
        }(i);
    }
    wg.Wait();
}

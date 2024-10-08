package ml

import (
    "fmt"
    "sync"
    "os"
    "encoding/json"
)

type FFNN struct {
    Shape []int

    Layers []*DenseLayer
}

/*
Create a new FFNN with requested shape.

Arguments:
    - shape ([]int): The shape of the network.

Returns:
    - *FFNN: The create network.
    - error: The error if any occured.

Example:
    n := NewFFNN([]int{3, 3, 2});
*/
func NewFFNN(shape []int) (*FFNN, error) {
    if len(shape) < 2 {
        return nil, fmt.Errorf("Error: Invalid network dimensions.");
    }

    var err error;
    layers := make([]*DenseLayer, len(shape) - 1);
    for i := 0; i < len(shape) - 1; i++ {
        layers[i], err = NewDenseLayer(shape[i], shape[i + 1]);
        if err != nil {
            return nil, fmt.Errorf("Error: Failed to create network DenseLayer. %w", err);
        }
    }

    return &FFNN{
        Shape: shape,
        Layers: layers,
    }, nil;
}

/*
Forward a batch of inputs through the network.

Arguments:
    - inputs ([][]float64): The input batch.

Returns:
    - [][][]float64: The results of the forward batch where outputs[layer][set][output].
    - error: The error if any occured.

Example:
    outputs, err := n.Forward(inputs);
*/
func (n *FFNN) Forward(inputs [][]float64) ([][][]float64, error) {
    if len(inputs) < 1 || len(inputs[0]) != n.Shape[0] {
        return nil, fmt.Errorf("Error: Invalid input dimension to forward.");
    }

    var err error;
    outputs := make([][][]float64, len(n.Layers));
    for i := 0; i < len(n.Layers); i++ {
        if i == 0 {
            outputs[i], err = n.Layers[i].Forward(inputs);
            if err != nil {
                return nil, fmt.Errorf("Error: Failed to foward through DenseLayer. %w", err);
            }
        } else {
            outputs[i], err = n.Layers[i].Forward(outputs[i - 1]);
            if err != nil {
                return nil, fmt.Errorf("Error: Failed to foward through DenseLayer. %w", err);
            }
        }
    }

    return outputs, nil;
}

/*
Do a backward pass through the network.

Arguments:
    - inputs ([][]float64): The inital inputs given to the network.
    - correct ([][]float64): The target outputs for the bacth.
    - outputs ([][][]float64): The output from a FFNN.Forward() pass.
    - lr float64: The learning rate.

Returns:
    - error: The error that occured if any.

Example:
    n.Backward(inputs, correct, outputs, 0.001):
*/
func (n *FFNN) Backward(
    inputs [][]float64, 
    correct [][]float64, 
    outputs [][][]float64, 
    lr float64,
) error {
    if (len(inputs) < 1 ||
        len(inputs) != len(correct) ||
        len(outputs) != len(n.Layers) || 
        len(outputs[0]) != len(inputs) ||
        len(inputs[0]) != n.Shape[0] ||
        len(correct[0]) != n.Shape[len(n.Layers)] ||
        len(outputs[len(n.Layers) - 1][0]) != n.Shape[len(n.Layers)]) {
        return fmt.Errorf("Error: Invalid dimensions on parameters for backward.");
    }

    var err error;
    errors := make([][][]float64, len(n.Layers));
    for i := len(n.Layers) - 1; i >= 0; i-- {
        if i == 0 {
            _, err = n.Layers[i].Backward(inputs, errors[i + 1], lr);
            if err != nil {
                return fmt.Errorf("Error: Failed doing backward. %w", err);
            }
        } else if i == len(n.Layers) - 1 {
            de := make([][]float64, len(inputs));
            var wg sync.WaitGroup;
            wg.Add(len(inputs))
            for j := 0; j < len(inputs); j++ {
                de[j] = make([]float64, n.Shape[len(n.Layers)]);
                go func(set int) {
                    defer wg.Done();
                    for k := 0; k < n.Shape[len(n.Layers)]; k ++ {
                        de[set][k] = outputs[i][set][k] - correct[set][k];
                    }
                }(j);
            }
            wg.Wait();

            errors[i], err = n.Layers[i].Backward(outputs[i - 1], de, lr);
            if err != nil {
                return fmt.Errorf("Error: Failed doing backward. %w", err);
            }
        } else {
            errors[i], err = n.Layers[i].Backward(outputs[i - 1], errors[i + 1], lr);
            if err != nil {
                return fmt.Errorf("Error: Failed doing backward. %w", err);
            }
        }
    }

    return nil;
}

/*
Applies backward deltas to all layers.

Arguments:
    - N/A

Returns:
    - N/A

Example:
    n.Flush();
*/
func (n *FFNN) Flush() {
    for i := 0; i < len(n.Layers); i++ {
        n.Layers[i].Flush();
    }
}

/*
Train the network with given parameters.

Arguments:
    - inputs ([][]float64): The batch input.
    - targets ([][]float64): The batch targets.
    - lr (float64): The learning rate.
    - iter (int): The number of iterations to train on.
    - cs (int): The chunk size.

Returns:
    - error: The error that occured if any.

Example:
    n.train(inputs, targets, 0.1, 100);
*/
func (n *FFNN) Train(inputs [][]float64, targets [][]float64, lr float64, iter int, cs int) error {
    if (len(inputs) < 1 || 
        len(inputs) != len(targets) ||
        len(inputs[0]) != n.Shape[0] ||
        len(targets[0]) != n.Shape[len(n.Layers)]) {
        return fmt.Errorf("Error: Invalid dimensions on parameters for train.");
    }

    chunks := int(len(inputs) / cs);
    if len(inputs) % cs > 0 {
        chunks += 1;
    }

    for i := 0; i < iter; i++ {
        for j := 0; j < chunks; j++ {
            start := j * cs;
            end := start + cs;
            if j == chunks - 1 {
                end = len(inputs);
            }

            outputs, err := n.Forward(inputs[start:end]);
            if err != nil {
                return fmt.Errorf("Error: Failed to forward pass. %w", err);
            }

            err = n.Backward(inputs[start:end], targets[start:end], outputs, lr);
            if err != nil {
                return fmt.Errorf("Error: Failed to backward pass. %w", err);
            }
            n.Flush();
        }
    }

    return nil;
}

/*
Saves the FFNN to a file in JSON format.

Arguments:
    - p (string): The path to save the model to.

Returns:
    - error: The error if any occured

Example:
    err := n.Save("./model.json");
*/
func (n *FFNN) Save(p string) error {
    f, err := os.Create(p);
    if err != nil {
        return fmt.Errorf("Error: Failed to create file %s. %w", p, err);
    }
    defer f.Close();
    
    e := json.NewEncoder(f);
    err = e.Encode(n);
    if err != nil {
        return fmt.Errorf("Error: Failed to save json. %w", err);
    }

    return nil;
}

/*
Create a new FFNN from saved model data.

Arguments:
    - p (string): The path to load the model from.

Returns:
    - error: The error if any occured

Example:
    n, err := n.LoadFFNN("./model.json");
    
*/
func LoadFFNN(p string) (*FFNN, error) {
    f, err := os.Open(p);
    if err != nil {
        return nil, fmt.Errorf("Error: Failed to open file %s. %w", p, err);
    }
    defer f.Close();
    
    var result FFNN;
    d := json.NewDecoder(f);
    d.Decode(&result);

    return &result, nil;
}

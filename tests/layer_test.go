package tests;

import (
    "testing"

    "github.com/MagnusChase03/GoML/ml"
)

func TestNewDenseLayer(t *testing.T) {
    _, err := ml.NewDenseLayer(1, 0);
    if err == nil {
        t.Errorf("Error: Dense layer of invalid dimensions created.");
    }

    _, err = ml.NewDenseLayer(0, 1);
    if err == nil {
        t.Errorf("Error: Dense layer of invalid dimensions created.");
    }

    d, err := ml.NewDenseLayer(3, 2);
    if err != nil {
        t.Errorf("Error: Dense layer of valid dimensions failed to be created. %v", err);
    }

    if d.Shape[0] != 3 || d.Shape[1] != 2 {
        t.Errorf("Error: Dense layer shape incorrect.");
    }

    if len(d.Weights) != 2 || len(d.Delta) != 2 || len(d.Weights[0]) != 3 || len(d.Delta[0]) != 3 {
        t.Errorf("Error: Dense layer weights/deltas matrix of incorrect shape.")
    }

    if len(d.Bias) != 2 || len(d.DeltaBias) != 2 {
        t.Errorf("Error: Dense layer bias/deltaBias matrix of incorrect shape.")
    }
}

func TestForward(t *testing.T) {
    d, err := ml.NewDenseLayer(3, 2);
    if err != nil {
        t.Errorf("Error: Dense layer of valid dimensions failed to be created. %v", err);
    }

    _, err = d.Forward([][]float64{});
    if err == nil {
        t.Errorf("Error: Dense layer forwarded empty inputs.");
    }

    _, err = d.Forward([][]float64{{1.0}});
    if err == nil {
        t.Errorf("Error: Dense layer forwarded invalid input dimensions.");
    }

    out, err := d.Forward([][]float64{{1.0, -2.0, 3.0}, {3.0, 2.0, -1.0}}); 
    if err != nil {
        t.Errorf("Error: Dense layer forwarded invalid input dimensions. %v", err);
    }

    if len(out) != 2 {
        t.Errorf("Error: Dense layer forwarded returned incorrent number of results.");
    }
}

func TestBackward(t *testing.T) {
    d, err := ml.NewDenseLayer(3, 2);
    if err != nil {
        t.Errorf("Error: Dense layer of valid dimensions failed to be created. %v", err);
    }

    inputs := [][]float64{{1.0, -2.0, 3.0}, {3.0, 2.0, -1.0}};
    out, err := d.Forward(inputs); 
    if err != nil {
        t.Errorf("Error: Dense layer forwarded invalid input dimensions. %v", err);
    }

    de, err := d.Backward(inputs, out, 0.001);
    if err != nil {
        t.Errorf("Error: Dense layer failed backward with valid inputs. %v", err);
    }

    if len(de) != 2 || len(de[0]) != 3 {
        t.Errorf("Error: Dense layer backward returned incorrect number of results for de.");
    }
}

func TestFlush(t *testing.T) {
    d, err := ml.NewDenseLayer(3, 2);
    if err != nil {
        t.Errorf("Error: Dense layer of valid dimensions failed to be created. %v", err);
    }

    inputs := [][]float64{{1.0, -2.0, 3.0}, {3.0, 2.0, -1.0}};
    out, err := d.Forward(inputs); 
    if err != nil {
        t.Errorf("Error: Dense layer forwarded invalid input dimensions. %v", err);
    }

    _, err = d.Backward(inputs, [][]float64{{0.5, -0.5}, {0.5, -0.5}}, 0.01);
    if err != nil {
        t.Errorf("Error: Dense layer failed backward with valid inputs. %v", err);
    }
    d.Flush();

    out2, err := d.Forward(inputs); 
    if err != nil {
        t.Errorf("Error: Dense layer forwarded invalid input dimensions. %v", err);
    }

    for i := 0; i < len(out); i++ {
        if out2[i][0] > out[i][0] || out2[i][1] < out[i][1] {
            t.Errorf("Error: Dense layer training iteration went in wrong direction.");
        }
    }
}

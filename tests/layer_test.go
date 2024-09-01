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
        t.Errorf("Error: Dense layer of valid dimensions failed to be created.");
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

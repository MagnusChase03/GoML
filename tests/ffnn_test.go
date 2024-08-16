package tests

import (
    "testing"

    "github.com/MagnusChase03/GoML/ml"
)

func TestNewFFNN(t *testing.T) {
    nn, err := ml.NewFFNN([]int{3, 2, 1})

    if err != nil {
        TestLog("ml.NewFFNN()", "Failed to create ffnn.", t)
    }

    if len(nn.Data) != 3 || len(nn.Bias) != 3 || len(nn.Weights) != 2 {
        TestLog("ml.NewFFNN()", "FFNN has incorrect number of layers.", t)
    }

    if len(nn.Data[0].Data) != 3 || len(nn.Data[1].Data) != 2 || len(nn.Data[2].Data) != 1 {
        TestLog("ml.NewFFNN()", "FFNN data layer not allocated properly.", t)
    }

    if len(nn.Bias[0].Data) != 3 || len(nn.Bias[1].Data) != 2 || len(nn.Bias[2].Data) != 1 {
        TestLog("ml.NewFFNN()", "FFNN bias layer not allocated properly.", t)
    }

    if len(nn.Weights[0].Data) != 3 || len(nn.Weights[0].Data[0]) != 2 {
        TestLog("ml.NewFFNN()", "FFNN weight layer not allocated properly.", t)
    }

    if len(nn.Weights[1].Data) != 2 || len(nn.Weights[1].Data[0]) != 1 {
        TestLog("ml.NewFFNN()", "FFNN weight layer not allocated properly.", t)
    }
}

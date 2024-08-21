package tests

import (
	"testing"

	"github.com/MagnusChase03/GoML/linalg"
	"github.com/MagnusChase03/GoML/ml"
)

func TestNewFFNN(t *testing.T) {
    nn, err := ml.NewFFNN([]int{3, 2, 1})
    if err != nil {
        TestLog("ml.NewFFNN()", "Failed to create ffnn.", t)
    }

    if len(nn.Bias) != 2 || len(nn.Weights) != 2 {
        TestLog("ml.NewFFNN()", "FFNN has incorrect number of layers.", t)
    }

    if nn.Bias[0].Rows != 2 || nn.Bias[1].Rows != 1 {
        TestLog("ml.NewFFNN()", "FFNN bias layer not allocated properly.", t)
    }

    if nn.Weights[0].Rows != 2 || nn.Weights[0].Cols != 3 {
        TestLog("ml.NewFFNN()", "FFNN weight layer not allocated properly.", t)
    }

    if nn.Weights[1].Rows != 1 || nn.Weights[1].Cols != 2 {
        TestLog("ml.NewFFNN()", "FFNN weight layer not allocated properly.", t)
    }
}

func TestFFNNForward(t *testing.T) {
    nn, err := ml.NewFFNN([]int{3, 2, 1})
    if err != nil {
        TestLog("ml.NewFFNN()", "Failed to create ffnn.", t)
    }

    i, err := linalg.VectorFromSlice([]float64{1.0, 2.0, 3.0})
    if err != nil {
        TestLog("ml.NewFFNN()", "Failed to create vector from slice.", t)
    }

    i2, err := linalg.VectorFromSlice([]float64{4.0, 5.0, 6.0})
    if err != nil {
        TestLog("ml.NewFFNN()", "Failed to create vector from slice.", t)
    }

    inputs := []*linalg.Matrix{i, i2}
    _, err = nn.Forward(inputs)
    if err != nil {
        TestLog("ml.NewFFNN()", "Failed to complete the forward function.", t)
    }
}

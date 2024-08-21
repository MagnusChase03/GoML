package tests

import (
    "context"
    "testing"

    "github.com/MagnusChase03/GoML/linalg"
)

func TestNewMatrix(t *testing.T) {
    m, err := linalg.NewMatrix(3, 2)
    if err != nil {
        TestLog("linalg.NewMatrix()", "Invalid return from matrix creation.", t)
    }

    _, err = linalg.NewMatrix(0, -1)
    if err == nil {
        TestLog("linalg.NewMatrix()", "Invalid return from matrix creation of invalid dimensions.", t)
    }

    if m.Rows != 3 || m.Cols != 2 {
        TestLog("linalg.NewMatrix()", "Incorrect row or col value.", t)
    }

    if len(m.Data) != 3 || len(m.Data[0]) != 2 {
        TestLog("linalg.NewMatrix()", "Incorrect row or col allocation.", t)
    }

    for i := 0; i < 3; i++ {
        for j := 0; j < 2; j++ {
            if m.Data[i][j] != 0 {
                TestLog("linalg.NewMatrix()", "Value in matrix is non-zero.", t)
            }
        }
    }
}

func TestVectorFromSlice(t *testing.T) {
    _, err := linalg.VectorFromSlice([]float64{})
    if err == nil {
        TestLog("linalg.VectorFromSlice()", "Dimension check failed.", t)
    }

    v, err := linalg.VectorFromSlice([]float64{1.0, 2.0, 3.0})
    if err != nil {
        TestLog("linalg.VectorFromSlice()", "Failed to create vector.", t)
    }

    if v.Cols != 1 || v.Rows != 3 {
        TestLog("linalg.VectorFromSlice()", "Failed to create vector with correct dimensions.", t)
    }

    if v.Data[0][0] != 1.0 || v.Data[1][0] != 2.0 || v.Data[2][0] != 3.0 {
        TestLog("linalg.VectorFromSlice()", "Failed to create vector with correct values.", t)
    }
}

func TestMatrixFromSlice(t *testing.T) {
    _, err := linalg.MatrixFromSlice([][]float64{})
    if err == nil {
        TestLog("linalg.MatrixFromSlice()", "Dimension check failed.", t)
    }

    m, err := linalg.MatrixFromSlice([][]float64{{1.0, 2.0}, {3.0, 4.0}})
    if err != nil {
        TestLog("linalg.MatrixFromSlice()", "Failed to create matrix.", t)
    }

    if m.Cols != 2 || m.Rows != 2 {
        TestLog("linalg.MatrixFromSlice()", "Failed to create matrix with correct dimensions.", t)
    }

    if m.Data[0][0] != 1.0 || m.Data[0][1] != 2.0 || m.Data[1][0] != 3.0 || m.Data[1][1] != 4.0 {
        TestLog("linalg.VectorFromSlice()", "Failed to create matrix with correct values.", t)
    }
}

func TestMatrixMultiply(t *testing.T) {
    m, _ := linalg.NewMatrix(2, 2)
    m2, _ := linalg.NewMatrix(2, 3)
    m3, _ := linalg.NewMatrix(1, 1)
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    if _, err := m.Multiply(ctx, m3); err == nil {
        TestLog("linalg.Matrix.Multiply()", "Dimension check failed.", t)
    }

    m.Data[0][0] = 1
    m.Data[0][1] = 1
    m2.Data[0][0] = 2
    m2.Data[1][0] = 2
    m2.Data[0][2] = 3
    m2.Data[1][2] = 3

    m4, err := m.Multiply(ctx, m2)
    if err != nil {
        TestLog("linalg.Matrix.Multiply()", "Dimension check failed for correct dimensions.", t)
    }

    if m4.Rows != 2 || m4.Cols != 3 {
        TestLog("linalg.Matrix.Multiply()", "Result matrix dimensions wrong.", t)
    }

    if m4.Data[0][0] != 4 || m4.Data[0][2] != 6 {
        TestLog("linalg.Matrix.Multiply()", "Matrix multiplication incorrect.", t)
    }
}

func TestMatrixAdd(t *testing.T) {
    m, _ := linalg.NewMatrix(2, 2)
    m2, _ := linalg.NewMatrix(2, 2)
    m3, _ := linalg.NewMatrix(1, 1)
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    if _, err := m.Add(ctx, m3); err == nil {
        TestLog("linalg.Matrix.Add()", "Dimension check failed.", t)
    }

    m.Data[0][0] = 1
    m.Data[0][1] = 1
    m2.Data[0][0] = 2
    m2.Data[1][0] = 2

    m4, err := m.Add(ctx, m2)
    if err != nil {
        TestLog("linalg.Matrix.Add()", "Dimension check failed for correct dimensions.", t)
    }

    if m4.Rows != 2 || m4.Cols != 2 {
        TestLog("linalg.Matrix.Add()", "Result matrix dimensions wrong.", t)
    }

    if m4.Data[0][0] != 3 || m4.Data[0][1] != 1 || m4.Data[1][0] != 2 {
        TestLog("linalg.Matrix.Add()", "Matrix addition incorrect.", t)
    }
}

func TestMatrixRelu(t *testing.T) {
    m, _ := linalg.NewMatrix(3, 1)
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    m.Data[0][0] = 1.0
    m.Data[1][0] = 2.0
    m.Data[2][0] = -1.0

    m2, err := m.Relu(ctx)
    if err != nil {
        TestLog("linalg.Matrix.Relu()", "ReLU function failed.", t)
    }

    if m2.Data[0][0] != 1.0 || m2.Data[1][0] != 2.0 || m2.Data[2][0] != 0.0 {
        TestLog("linalg.Matrix.Relu()", "ReLU function failed in its calculation.", t)
    }

}

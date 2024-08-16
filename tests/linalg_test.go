package tests

import (
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

func TestMatrixMultiply(t *testing.T) {
    m, _ := linalg.NewMatrix(2, 2)
    m2, _ := linalg.NewMatrix(2, 3)
    m3, _ := linalg.NewMatrix(1, 1)

    if _, err := m.Multiply(m3); err == nil {
        TestLog("linalg.Matrix.Multiply()", "Dimension check failed.", t)
    }

    m.Data[0][0] = 1
    m.Data[0][1] = 1
    m2.Data[0][0] = 2
    m2.Data[1][0] = 2
    m2.Data[0][2] = 3
    m2.Data[1][2] = 3

    m4, err := m.Multiply(m2)
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

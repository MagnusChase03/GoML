package linalg

import (
    "fmt"
    "sync"
)

type Matrix struct {
    Data [][]float64
    Rows int
    Cols int
};

func NewMatrix(r int, c int) *Matrix {
    if r < 1 {
        r = 1
    }
    if c < 1 {
        c = 1
    }

    data := make([][]float64, r)
    for i := 0; i < r; i++ {
        data[i] = make([]float64, c)
    }

    return &Matrix{
        Data: data,
        Rows: r,
        Cols: c,
    }
}

func (m *Matrix) Multiply(m2 *Matrix) (*Matrix, error) {
    if m.Cols != m2.Rows {
        return nil, fmt.Errorf("Matrix multiply dimension error.");
    }

    r := NewMatrix(m.Rows, m2.Cols)

    var wg sync.WaitGroup
    wg.Add(m.Rows)
    for i := 0; i < m.Rows; i++ {
        go func(row int) {
            defer wg.Done()
            for j := 0; j < m2.Rows; j++ {
                for k := 0; k < m2.Cols; k++ {
                    r.Data[row][k] += m.Data[row][j] * m2.Data[j][k]
                }
            }
        }(i)
    }

    wg.Wait()
    return r, nil
}

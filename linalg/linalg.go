package linalg

import (
	"fmt"
	"math/rand"
	"sync"
)

type Matrix struct {
    Data [][]float64
    Rows int
    Cols int
};

func NewMatrix(r int, c int) (*Matrix, error) {
    if r < 1 || c < 1 {
        return nil, fmt.Errorf("Cannot create matrix of dimension %dx%d.", r, c)
    }

    data := make([][]float64, r)
    for i := 0; i < r; i++ {
        data[i] = make([]float64, c)
    }

    return &Matrix{
        Data: data,
        Rows: r,
        Cols: c,
    }, nil
}

func NewRandomMatrix(r int, c int) (*Matrix, error) {
    if r < 1 || c < 1 {
        return nil, fmt.Errorf("Cannot create matrix of dimension %dx%d.", r, c)
    }

    data := make([][]float64, r)
    var wg sync.WaitGroup
    wg.Add(r)

    for i := 0; i < r; i++ {
        go func(row int) {
            defer wg.Done()
            data[i] = make([]float64, c)
            for j := 0; j < c; j++ {
                data[i][j] = rand.Float64() / 10.0
            }
        }(i)
    }

    wg.Wait()
    return &Matrix{
        Data: data,
        Rows: r,
        Cols: c,
    }, nil
}

func (m *Matrix) Multiply(m2 *Matrix) (*Matrix, error) {
    if m.Cols != m2.Rows {
        return nil, fmt.Errorf("Matrix multiply dimension error %d != %d.", m.Cols, m2.Rows);
    }

    r, err := NewMatrix(m.Rows, m2.Cols)
    if err != nil {
        return nil, fmt.Errorf("%w - Failed to create result matrix.", err)
    }

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

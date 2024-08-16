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

func VectorFromSlice(d []float64) (*Matrix, error) {
    if len(d) < 1 {
        return nil, fmt.Errorf("Cannot create matrix of dimension %dx1.", len(d))
    }

    data := make([][]float64, len(d))
    for i := 0; i < len(d); i++ {
        data[i] = make([]float64, 1)
        data[i][0] = d[i]
    }

    return &Matrix{
        Data: data,
        Rows: len(d),
        Cols: 1,
    }, nil
}

func MatrixFromSlice(d [][]float64) (*Matrix, error) {
    if len(d) < 1 || len(d[0]) < 1 {
        return nil, fmt.Errorf("Cannot create matrix of dimension %dx%d.", len(d), len(d[0]))
    }

    return &Matrix{
        Data: d,
        Rows: len(d),
        Cols: len(d[0]),
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

func (m *Matrix) Add(m2 *Matrix) (*Matrix, error) {
    if m.Cols != m2.Cols || m.Rows != m2.Rows {
        return nil, fmt.Errorf("Matrix addition dimension error %d != %d.", m.Cols, m2.Rows);
    }

    r, err := NewMatrix(m.Rows, m.Cols)
    if err != nil {
        return nil, fmt.Errorf("%w - Failed to create result matrix.", err)
    }

    var wg sync.WaitGroup
    wg.Add(m.Rows)
    for i := 0; i < m.Rows; i++ {
        go func(row int) {
            defer wg.Done()
            for j := 0; j < m.Cols; j++ {
                r.Data[row][j] = m.Data[row][j] + m2.Data[row][j]
            }
        }(i)
    }

    wg.Wait()
    return r, nil
}

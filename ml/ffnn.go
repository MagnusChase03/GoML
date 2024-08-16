package ml

import (
	"fmt"

	"github.com/MagnusChase03/GoML/linalg"
)

type FFNN struct {
    Weights []*linalg.Matrix
    Bias []*linalg.Matrix
    Data []*linalg.Matrix
}

func NewFFNN(shape []int) (*FFNN, error) {
    w := make([]*linalg.Matrix, len(shape) - 1)
    b := make([]*linalg.Matrix, len(shape))
    d := make([]*linalg.Matrix, len(shape))
    errChan := make(chan error)

    go func(e chan error) {
        var err error
        for i := 0; i < len(shape); i++ {
            b[i], err = linalg.NewMatrix(shape[i], 1)
            if err != nil {
                e<-fmt.Errorf("%w - Failed to create bias matrix.", err)
                return
            }
        }
        e<-nil
    }(errChan)

    go func(e chan error) {
        var err error
        for i := 0; i < len(shape); i++ {
            d[i], err = linalg.NewMatrix(shape[i], 1)
            if err != nil {
                e<-fmt.Errorf("%w - Failed to create data matrix.", err)
                return
            }
        }
        e<-nil
    }(errChan)

    go func(e chan error) {
        var err error
        for i := 0; i < len(shape) - 1; i++ {
            w[i], err = linalg.NewRandomMatrix(shape[i], shape[i + 1])
            if err != nil {
                e<-fmt.Errorf("%w - Failed to create weight matrix.", err)
                return
            }
        }
        e<-nil
    }(errChan)

    for i := 0; i < 3; i++ {
        select {
            case err := <-errChan:
                if err != nil {
                    return nil, err
                }
        }
    }

    return &FFNN{
        Weights: w,
        Bias: b,
        Data: d,
    }, nil
}

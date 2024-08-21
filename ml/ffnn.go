package ml

import (
	"fmt"
    "context"

	"github.com/MagnusChase03/GoML/linalg"
)

type FFNN struct {
    Weights []*linalg.Matrix
    Bias []*linalg.Matrix
}

func NewFFNN(shape []int) (*FFNN, error) {
    if len(shape) < 2 {
        return nil, fmt.Errorf("Failed to create FFNN with %d layers.", len(shape))
    }

    w := make([]*linalg.Matrix, len(shape) - 1)
    b := make([]*linalg.Matrix, len(shape) - 1)
    errChan := make(chan error)
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    go func(e chan error, c context.Context) {
        var err error
        for i := 0; i < len(shape) - 1; i++ {
            select {
            case <-c.Done():
                return
            default:
                b[i], err = linalg.NewMatrix(shape[i + 1], 1)
                if err != nil {
                    e<-fmt.Errorf("%w - Failed to create bias matrix.", err)
                    return
                }
            }
        }
        e<-nil
    }(errChan, ctx)

    go func(e chan error, c context.Context) {
        var err error
        for i := 0; i < len(shape) - 1; i++ {
            select {
            case <-c.Done():
                return
            default:
                w[i], err = linalg.NewRandomMatrix(shape[i + 1], shape[i])
                if err != nil {
                    e<-fmt.Errorf("%w - Failed to create weight matrix.", err)
                    return
                }
            }
        }
        e<-nil
    }(errChan, ctx)

    for i := 0; i < 2; i++ {
        select {
        case err := <-errChan:
            if err != nil {
                cancel()
                return nil, err
            }
        }
    }

    return &FFNN{
        Weights: w,
        Bias: b,
    }, nil
}

func (f *FFNN) Forward(inputs []*linalg.Matrix) ([][]*linalg.Matrix, error) {
    if len(inputs) < 1 || inputs[0].Rows != f.Weights[0].Cols || inputs[0].Cols != 1 {
        return nil, fmt.Errorf("Input shape is invalid for forward pass.")
    }

    errChan := make(chan error)
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    modelCache := make([][]*linalg.Matrix, len(inputs))
    for i := 0; i < len(inputs); i++ {
        modelCache[i] = make([]*linalg.Matrix, len(f.Weights) + 1)
    }

    for i := 0; i < len(inputs); i++ {
        go func(set int, e chan error, c context.Context) {
            var err error
            result := inputs[set]

            cache, err := inputs[set].Clone(c)
            if err != nil {
                errChan<-fmt.Errorf("%w - Failed to clone results to cache.", err)
                return
            }
            modelCache[set][0] = cache

            for j := 0; j < len(f.Weights); j++ {
                select {
                case <-c.Done():
                    return
                default:
                    result, err = f.Weights[j].Multiply(c, result)
                    if err != nil {
                        errChan<-fmt.Errorf("%w - Failed to multiply weights to result.", err)
                        return
                    }

                    result, err = result.Add(c, f.Bias[j])
                    if err != nil {
                        errChan<-fmt.Errorf("%w - Failed to add bias to result.", err)
                        return
                    }

                    if j < len(f.Weights) - 1 {
                        result, err = result.Relu(c)
                        if err != nil {
                            errChan<-fmt.Errorf("%w - Failed to use ReLU on result.", err)
                            return
                        }
                    }

                    cache, err = result.Clone(c)
                    if err != nil {
                        errChan<-fmt.Errorf("%w - Failed to clone results to cache.", err)
                        return
                    }
                    modelCache[set][j + 1] = cache
                }
            }
            errChan<-nil
        }(i, errChan, ctx)
    }

    for i := 0; i < len(inputs); i++ {
        select {
        case err := <-errChan:
            if err != nil {
                cancel()
                return nil, err
            }
        }
    }

    return modelCache, nil
}

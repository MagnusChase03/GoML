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
                w[i], err = linalg.NewRandomMatrix(shape[i], shape[i + 1])
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

func (f *FFNN) Forward(inputs []*linalg.Matrix) ([]*linalg.Matrix, error) {
    if len(inputs) < 1 || len(inputs[0].Data) != len(f.Weights[0].Data[0]) {
        return nil, fmt.Errorf("Input shape is invalid for forward pass.")
    }

    errChan := make(chan error)
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    outputs := make([]*linalg.Matrix, len(inputs))
    for i := 0; i < len(inputs); i++ {
        go func(set int, e chan error, c context.Context) {
            var err error
            result := inputs[set]
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
                }
            }
            outputs[set] = result
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

    return outputs, nil
}

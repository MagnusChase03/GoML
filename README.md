# GoML

**Small library for a simple and fast FFNN in Go.**

## Usage

To run the example program:

`$ go run main.go`

## API

### linalg

```go
type Matrix struct {
    Data [][]float64
    Rows int
    Cols int
};
```

`NewMatrix(r int, c int) *Matrix`

Returns a new matrix of size **r**x**c**.

`(m *Matrix) Multiply(m2 *Matrix) (*Matrix, error)`

Multiply two matrices which will get returned unless there was an error.

### ml

# GoML

*A simple and fast feed forward neural network library.*

## Installation

To build the example program:

```
$ go build main.go
```

## Usage

To run the example program:

```
$ ./main
```

To use in another Go project, simply import the library:

```
import "github.com/MagnusChase03/GoML/ml"

...

n, err := ml.NewFFNN([]int{3, 2, 1});
```

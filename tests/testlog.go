package tests

import (
    "testing"
)

func TestLog(funcName string, message string, t *testing.T) {
    t.Errorf("[%s] %s", funcName, message) 
}

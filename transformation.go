package leaves

import (
	"fmt"

	"github.com/dmitryikh/leaves/util"
)

type Transform interface {
	Transform(rawPredictions []float64, outputPredictions []float64, startIndex int) error
	NOutputGroups() int
	Type() TransformType
	Name() string
}

type TransformType int

const (
	Raw      TransformType = 0
	Logistic TransformType = 1
	Softmax  TransformType = 2
)

func (t TransformType) Name() string {
	transformNames := [...]string {
		"raw",
		"logistic",
		"softmax",
	}
	if t < Raw || t > Softmax {
		return "unknown"
	}

	return transformNames[t]
}

type TransformRaw struct {
	nOutputGroups int
}

func (t *TransformRaw) Transform(rawPredictions []float64, outputPredictions []float64, startIndex int) error {
	for i, v := range rawPredictions {
		outputPredictions[startIndex + i] = v
	}
	return nil
}

func (t *TransformRaw) NOutputGroups() int {
	return t.nOutputGroups
}

func (t *TransformRaw) Type() TransformType {
	return Raw
}

func (t *TransformRaw) Name() string {
	return Raw.Name()
}

type TransformLogistic struct {}

func (t *TransformLogistic) Transform(rawPredictions []float64, outputPredictions []float64, startIndex int) error {
	if len(rawPredictions) != 1 {
		return fmt.Errorf("expected len(rawPredictions) = 1 (got %d)", len(rawPredictions))
	}

	outputPredictions[startIndex] = util.Sigmoid(rawPredictions[0])
	return nil
}

func (t *TransformLogistic) NOutputGroups() int {
	return 1
}

func (t *TransformLogistic) Type() TransformType {
	return Logistic
}

func (t *TransformLogistic) Name() string {
	return Logistic.Name()
}

type TransformSoftmax struct {
	nClasses int
}

func (t *TransformSoftmax) Transform(rawPredictions []float64, outputPredictions []float64, startIndex int) error {
	if len(rawPredictions) != t.nClasses {
		return fmt.Errorf("expected len(rawPredictions) = %d (got %d)", t.nClasses, len(rawPredictions))
	}

	util.SoftmaxFloat64Slice(rawPredictions, outputPredictions, startIndex)
	return nil
}

func (t *TransformSoftmax) NOutputGroups() int {
	return t.nClasses
}

func (t *TransformSoftmax) Type() TransformType {
	return Softmax
}

func (t *TransformSoftmax) Name() string {
	return Softmax.Name()
}
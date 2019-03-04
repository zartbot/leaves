package leaves

import (
	"errors"
)

func LoadModel(filename string, modelType string) (*Ensemble, error) {
	switch modelType {
	case "lightgbm":
		return LGEnsembleFromFile(filename)
	case "xgboost":
		return XGEnsembleFromFile(filename)
	case "xgboost-linear":
		return XGBLinearFromFile(filename)
	case "sklearn":
		return SKEnsembleFromFile(filename)
	}
	return nil, errors.New("Invalid model type")
}

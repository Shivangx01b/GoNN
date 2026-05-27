package data

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"

	"gonn/tensor"
)

// LoadCSV loads a CSV file and splits it into features X and target y.
//
// If hasHeader is true the first row is skipped. targetCol is the column
// index (in the original row layout) that holds the target value; the
// remaining columns become features. A negative targetCol is interpreted
// from the end (e.g. -1 = last column).
//
// All cells must parse as float64. The returned shapes are X(N, F) and
// y(N,), where F = numCols - 1.
func LoadCSV(path string, hasHeader bool, targetCol int) (X *tensor.Tensor, y *tensor.Tensor, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("LoadCSV: opening %s: %w", path, err)
	}
	defer f.Close()

	rd := csv.NewReader(f)
	rd.FieldsPerRecord = -1 // tolerate variable widths during parse; we validate below

	if hasHeader {
		if _, err := rd.Read(); err != nil {
			if err == io.EOF {
				return nil, nil, fmt.Errorf("LoadCSV: %s is empty", path)
			}
			return nil, nil, fmt.Errorf("LoadCSV: reading header: %w", err)
		}
	}

	var rows [][]string
	numCols := -1
	for {
		row, err := rd.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, nil, fmt.Errorf("LoadCSV: reading row %d: %w", len(rows)+1, err)
		}
		if numCols < 0 {
			numCols = len(row)
		} else if len(row) != numCols {
			return nil, nil, fmt.Errorf("LoadCSV: row %d has %d cols, expected %d", len(rows)+1, len(row), numCols)
		}
		rows = append(rows, row)
	}
	if len(rows) == 0 {
		return nil, nil, fmt.Errorf("LoadCSV: no data rows in %s", path)
	}
	if numCols < 2 {
		return nil, nil, fmt.Errorf("LoadCSV: need at least 2 columns, got %d", numCols)
	}

	tCol := targetCol
	if tCol < 0 {
		tCol += numCols
	}
	if tCol < 0 || tCol >= numCols {
		return nil, nil, fmt.Errorf("LoadCSV: targetCol %d out of range [0,%d)", targetCol, numCols)
	}

	n := len(rows)
	feat := numCols - 1
	xData := make([]float64, n*feat)
	yData := make([]float64, n)
	for i, row := range rows {
		fi := 0
		for j, cell := range row {
			v, perr := strconv.ParseFloat(cell, 64)
			if perr != nil {
				return nil, nil, fmt.Errorf("LoadCSV: parsing row %d col %d (%q): %w", i+1, j, cell, perr)
			}
			if j == tCol {
				yData[i] = v
			} else {
				xData[i*feat+fi] = v
				fi++
			}
		}
	}

	return tensor.New(xData, n, feat), tensor.New(yData, n), nil
}

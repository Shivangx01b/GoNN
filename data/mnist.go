package data

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"gonn/tensor"
)

// LoadMNIST loads the four standard MNIST IDX files from the given
// directory. It looks for either the raw or gzipped names:
//
//	train-images-idx3-ubyte[.gz]
//	train-labels-idx1-ubyte[.gz]
//	t10k-images-idx3-ubyte[.gz]
//	t10k-labels-idx1-ubyte[.gz]
//
// Returned tensor shapes:
//
//	trainX (60000, 1, 28, 28), trainY (60000,)
//	testX  (10000, 1, 28, 28), testY  (10000,)
//
// Pixel values are scaled to [0, 1]. Labels are returned as float64.
// LoadMNIST never downloads anything; if a file is missing, a clear error
// is returned.
func LoadMNIST(path string) (trainX, trainY, testX, testY *tensor.Tensor, err error) {
	trainImgs, err := openMNISTFile(path, "train-images-idx3-ubyte")
	if err != nil {
		return nil, nil, nil, nil, err
	}
	trainLbls, err := openMNISTFile(path, "train-labels-idx1-ubyte")
	if err != nil {
		return nil, nil, nil, nil, err
	}
	testImgs, err := openMNISTFile(path, "t10k-images-idx3-ubyte")
	if err != nil {
		return nil, nil, nil, nil, err
	}
	testLbls, err := openMNISTFile(path, "t10k-labels-idx1-ubyte")
	if err != nil {
		return nil, nil, nil, nil, err
	}

	trainX, err = mnistFromIDX(trainImgs, true)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("LoadMNIST: parsing train images: %w", err)
	}
	trainY, err = mnistFromIDX(trainLbls, false)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("LoadMNIST: parsing train labels: %w", err)
	}
	testX, err = mnistFromIDX(testImgs, true)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("LoadMNIST: parsing test images: %w", err)
	}
	testY, err = mnistFromIDX(testLbls, false)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("LoadMNIST: parsing test labels: %w", err)
	}
	return trainX, trainY, testX, testY, nil
}

// openMNISTFile opens dir/base or dir/base.gz, returning a Reader over the
// (decompressed) contents.
func openMNISTFile(dir, base string) (io.Reader, error) {
	raw := filepath.Join(dir, base)
	gz := raw + ".gz"

	if f, err := os.Open(raw); err == nil {
		// Read fully into memory; MNIST files are small enough.
		buf, rerr := io.ReadAll(f)
		_ = f.Close()
		if rerr != nil {
			return nil, fmt.Errorf("LoadMNIST: reading %s: %w", raw, rerr)
		}
		return strings.NewReader(string(buf)), nil
	}
	if f, err := os.Open(gz); err == nil {
		zr, zerr := gzip.NewReader(f)
		if zerr != nil {
			_ = f.Close()
			return nil, fmt.Errorf("LoadMNIST: gzip header in %s: %w", gz, zerr)
		}
		buf, rerr := io.ReadAll(zr)
		_ = zr.Close()
		_ = f.Close()
		if rerr != nil {
			return nil, fmt.Errorf("LoadMNIST: decompressing %s: %w", gz, rerr)
		}
		return strings.NewReader(string(buf)), nil
	}

	return nil, fmt.Errorf("LoadMNIST: could not find %s or %s (downloads are disabled)", raw, gz)
}

// mnistFromIDX parses an IDX-formatted byte stream. If isImage is true the
// result is a (N, 1, rows, cols) float64 tensor scaled to [0, 1];
// otherwise the result is a (N,) float64 label tensor.
//
// IDX format:
//
//	0000:0001  magic byte (zero)
//	0002       magic byte (zero)
//	0003       data type (0x08 = unsigned byte)
//	0004       number of dimensions
//	then big-endian uint32 sizes, then raw data.
func mnistFromIDX(r io.Reader, isImage bool) (*tensor.Tensor, error) {
	var magic uint32
	if err := binary.Read(r, binary.BigEndian, &magic); err != nil {
		return nil, fmt.Errorf("idx: reading magic: %w", err)
	}
	// magic layout: 0x00 0x00 type ndim
	dtype := byte((magic >> 8) & 0xff)
	ndim := int(magic & 0xff)
	if (magic>>16) != 0 {
		return nil, fmt.Errorf("idx: invalid magic 0x%08x (expected leading zero bytes)", magic)
	}
	if dtype != 0x08 {
		return nil, fmt.Errorf("idx: unsupported data type 0x%02x (only unsigned byte 0x08 is supported)", dtype)
	}
	if isImage && ndim != 3 {
		return nil, fmt.Errorf("idx: expected 3-D image data, got ndim=%d", ndim)
	}
	if !isImage && ndim != 1 {
		return nil, fmt.Errorf("idx: expected 1-D label data, got ndim=%d", ndim)
	}

	dims := make([]uint32, ndim)
	for i := 0; i < ndim; i++ {
		if err := binary.Read(r, binary.BigEndian, &dims[i]); err != nil {
			return nil, fmt.Errorf("idx: reading dim %d: %w", i, err)
		}
	}

	total := 1
	for _, d := range dims {
		total *= int(d)
	}
	bytesBuf := make([]byte, total)
	if _, err := io.ReadFull(r, bytesBuf); err != nil {
		return nil, fmt.Errorf("idx: reading payload (%d bytes): %w", total, err)
	}

	if isImage {
		n := int(dims[0])
		rows := int(dims[1])
		cols := int(dims[2])
		data := make([]float64, total)
		for i, b := range bytesBuf {
			data[i] = float64(b) / 255.0
		}
		return tensor.New(data, n, 1, rows, cols), nil
	}

	n := int(dims[0])
	data := make([]float64, n)
	for i, b := range bytesBuf {
		data[i] = float64(b)
	}
	return tensor.New(data, n), nil
}

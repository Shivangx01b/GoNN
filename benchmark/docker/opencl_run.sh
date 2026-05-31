#!/usr/bin/env bash
# Build + verify GoNN's OpenCL backend. Runs INSIDE the gonn-cuda docker image.
#
#   docker run --rm --gpus all -v "$PWD":/work -w /work gonn-cuda \
#       bash benchmark/docker/opencl_run.sh
#
# This host's Docker/WSL2 GPU passthrough does NOT inject NVIDIA's OpenCL driver,
# so we verify correctness on a portable OpenCL runtime (oclgrind, fp64) through
# the standard Khronos ICD loader. The SAME -tags opencl binary runs on the GPU
# on any machine with a real GPU OpenCL ICD (native Linux + NVIDIA driver, or the
# Windows host where the NVIDIA driver provides OpenCL.dll).
set -e

echo "=== install Khronos ICD loader + oclgrind (fp64 OpenCL runtime) ==="
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq ocl-icd-libopencl1 ocl-icd-opencl-dev oclgrind >/dev/null 2>&1
mkdir -p /etc/OpenCL/vendors
echo /usr/lib/oclgrind/liboclgrind-rt-icd.so > /etc/OpenCL/vendors/oclgrind.icd

export PATH=/usr/local/go/bin:$PATH
# CL headers ship with CUDA; link against the system Khronos loader (it loads any
# registered vendor ICD, unlike NVIDIA's loader).
export CGO_CFLAGS="-I/usr/local/cuda/include"
export CGO_LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu"

echo "=== build -tags opencl ==="
go build -tags opencl ./...

echo "=== OpenCL correctness check (oclgrind fp64 device) ==="
go run -tags opencl ./benchmark/openclcheck
echo "=== done ==="

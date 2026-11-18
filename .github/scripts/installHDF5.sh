#!/usr/bin/env bash
set -euo pipefail

hdf5Version=1.14.6
buildDir=$(mktemp -d)
trap 'rm -rf "$buildDir"' EXIT

curl --fail --location --silent --show-error \
    "https://github.com/HDFGroup/hdf5/releases/download/hdf5_${hdf5Version}/hdf5-${hdf5Version}.tar.gz" \
    --output "$buildDir/hdf5.tar.gz"
tar -xzf "$buildDir/hdf5.tar.gz" -C "$buildDir"
cd "$buildDir/hdf5-${hdf5Version}"

./configure \
    --prefix=/tmp/consenrichHDF5 \
    --enable-shared \
    --disable-static \
    --disable-fortran \
    --disable-cxx \
    --disable-tests \
    --disable-tools

make -j2
make install

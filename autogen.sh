#!/usr/bin/bash

mkdir -p build
cd build
cmake ..
sudo make -j8
cd ..


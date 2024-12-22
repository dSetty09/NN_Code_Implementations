#!/bin/bash

: '
THIS SH FILE ONLY WORKS AS INTENDED IF YOU ARE IN
THE MAIN PROJECT DIRECTORY 
'

rm -rf build
mkdir build
cd build

cmake ..
make

cd ..

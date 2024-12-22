#!/bin/sh

: '
THIS SH FILE ONLY WORKS AS INTENDED IF YOU ARE IN
THE MAIN PROJECT DIRECTORY 
'

for test in ./build/test*; do
    echo "Running $test"
    $test
done

#!/bin/bash

: '
THIS SH FILE ONLY WORKS AS INTENDED IF YOU ARE IN
THE MAIN PROJECT DIRECTORY 
'

for running_script in ./run*; do
    git add $running_script
done

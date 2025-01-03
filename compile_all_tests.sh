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

echo
echo "-- Following running scripts have been created"

for test in ./build/test*; do
    name="run_$(basename $test)"
    
    echo "#!/bin/bash" > "$name.sh"
    echo >> "$name.sh"
    echo ": '" >> "$name.sh"
    echo "THIS SH FILE ONLY WORKS AS INTENDED IF YOU ARE IN" >> "$name.sh"
    echo "THE MAIN PROJECT DIRECTORY" >> "$name.sh"
    echo "'" >> "$name.sh"
    echo >> "$name.sh"
    echo "$test" >> "$name.sh"
    echo >> "$name.sh"

    chmod +x "$name.sh"

    echo -e "* \033[38;5;183m$name.sh\033[0m"
done

echo

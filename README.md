# Vedya Labs Neural Network Implementation Project

## Installing Google Test if Needed
**For macOS**  
1. **Using [Homebrew](https://brew.sh/)**:

```bash
brew install googletest
```
After installation, Homebrew provides the `gtest` library in its default paths like `/usr/local/opt/googletest`.
 
2. **Manual Installation** :

```bash
git clone https://github.com/google/googletest.git
cd googletest
cmake -S . -B build
cmake --build build
sudo cmake --install build
```


---

**For Linux**  
1. **Using Package Manager**  (Ubuntu/Debian):

```bash
sudo apt update
sudo apt install libgtest-dev
```
After installation, compile the source code (as `libgtest-dev` installs the source):

```bash
cd /usr/src/gtest
sudo cmake .
sudo make
sudo cp *.a /usr/lib
```
 
2. **Manual Installation** :

```bash
git clone https://github.com/google/googletest.git
cd googletest
cmake -S . -B build
cmake --build build
sudo cmake --install build
```


---

**For Windows**  
1. **Using vcpkg** : 
  - Install [vcpkg](https://github.com/microsoft/vcpkg)  if not already installed.
 
  - Install `gtest` using vcpkg:

```bash
vcpkg install gtest
```
 
2. **Manual Installation** : 
  - Clone the repository:

```bash
git clone https://github.com/google/googletest.git
cd googletest
```
 
  - Use CMake to generate project files and build:

```bash
cmake -S . -B build
cmake --build build
```

  - Add the compiled library paths to your project or system path as needed.


---

**Verifying Installation** After installing `gtest`, you can verify it by compiling and running a simple test program.**Example Test Program:** 

```cpp
#include <gtest/gtest.h>

TEST(SampleTest, AssertionTrue) {
    ASSERT_TRUE(true);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

Compile with:


```bash
g++ -std=c++17 -o test_sample test_sample.cpp -lgtest -lgtest_main -pthread
./test_sample
```

If the test runs successfully, your installation is complete!


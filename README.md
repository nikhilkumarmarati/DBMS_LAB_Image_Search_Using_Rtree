# DBMS_LAB_Image_Search_Using_Rtree

# --------------------------------------
# Setup and Run Instructions
# --------------------------------------

# 1. (Optional) Create and activate a virtual environment
python -m venv venv

# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# 2. Install required Python packages
pip install -r requirements.txt

# 3. Ensure C++ build tools are installed (required for pybind11)
#    - Linux/macOS: install build-essential
#    - Windows: install MSVC or mingw-w64 (e.g., via Chocolatey)

# 4. Download and extract the STL-10 binary dataset
mkdir -p stl10_binary
wget http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz
tar -xzvf stl10_binary.tar.gz -C stl10_binary

# (Windows users: download and extract manually into stl10_binary)

# 5. Build the rtree_image_search executable
make -f Makefile_rtree

# 6. Build the rtree.so Python extension using pybind11
c++ -O3 -Wall -shared -std=c++17 -fPIC \
  $(python3 -m pybind11 --includes) \
  rtree.cpp -o rtree.so \
  $(python3-config --ldflags)

# (Windows users: adapt this command using g++ or MSVC equivalent)

# --------------------------------------
# Usage
# --------------------------------------

# Extract features from images using DINOV2
./rtree_image_search extract stl10_binary

# Build the R-tree index
./rtree_image_search build stl10_binary

# Query using R-tree
./rtree_image_search query stl10_binary

# Launch the Python interface
python interface_rtree.py

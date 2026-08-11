# Build and run the functional testbench with plain g++ (no Vitis needed).
#
#   make            # build ./tb
#   make test       # build and run against data/ + golden/
#
# For Vitis HLS synthesis, add transformer.cpp/.h to an hls project, swap the
# typedefs in transformer.h for ap_int/ap_fixed, and add pragmas per board.

CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall

SRC = transformer.cpp params_io.cpp testbench.cpp
BIN = tb

$(BIN): $(SRC) transformer.h
	$(CXX) $(CXXFLAGS) $(SRC) -o $(BIN)

.PHONY: test clean
test: $(BIN)
	./$(BIN) data golden

clean:
	rm -f $(BIN)

# --- Compiler and Target Name ---
CXX = g++
TARGET = konane_GroupD
SRC = KonaneWithWeights.cpp

# --- Aggressive Optimization Flags ---
# -std=c++17   : Required for your C++17 structured bindings.
# -O3          : Maximum standard optimization level.
# -march=native: Tells the compiler to optimize for the exact CPU architecture of the Linux server.
# -flto        : Link-Time Optimization (analyzes the whole program for more speed).
# -pthread     : Required for std::thread, std::mutex, and std::future.
# -Wall -Wextra: Shows helpful warnings.
CXXFLAGS = -std=c++17 -O3 -march=native -flto -Wall -Wextra -pthread

# --- Linking Flags ---
LDFLAGS = -pthread -flto

# --- Build Rules ---
# Object files derived from source files
OBJ = $(SRC:.cpp=.o)

# Default rule when you just type 'make'
all: $(TARGET)

# Rule to link the object files into the final executable
$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Rule to compile the source code into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to clean up compiled files ('make clean')
clean:
	rm -f $(OBJ) $(TARGET)

# Phony targets to prevent conflicts with file names
.PHONY: all clean
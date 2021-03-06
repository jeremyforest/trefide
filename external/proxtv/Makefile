# Detect Operating System
ifeq ($(OS),Windows_NT)
    detected_OS := Windows
else
    detected_OS := $(shell sh -c 'uname -s 2>/dev/null || echo not')
endif

# Set Windows Specific Environment Variables: TODO
ifeq ($(detected_OS),Windows)
    echo "Installation on Windows not currently supported."
endif

# Set MacOS Specific Environment Variables
ifeq ($(detected_OS),Darwin)
    EXT=.dylib
    LDFLAGS = -dynamiclib
endif

# Set Linux Specific Environment Variables
ifeq ($(detected_OS),Linux)
    EXT=.so
    LDFLAGS = -shared
endif

# If undefined (ie. not using conda-build), set PREFIX to active env
PREFIX ?= $(CONDA_PREFIX)

# Project Structure Dependent Variables

# OS Agnostic Environment Variables
TARGET = $(PREFIX)/lib/libproxtv$(EXT)
LDLIBS = -lmkl_intel_lp64 -lmkl_core -lm
SRCS = condat_fast_tv.cpp johnsonRyanTV.cpp TVL1opt_kolmogorov.cpp LPopt.cpp TV2Dopt.cpp TV2DWopt.cpp TVgenopt.cpp TVL1opt.cpp TVL1opt_tautstring.cpp TVL1opt_hybridtautstring.cpp TVL1Wopt.cpp TVL2opt.cpp TVLPopt.cpp TVNDopt.cpp utils.cpp
OBJS = $(patsubst %.cpp,%.o,$(SRCS))

INCLUDES = -I $(PREFIX)/include
LDFLAGS += -L $(PREFIX)/lib

# Compiler Dependent Environment Variables

ifeq ($(CXX),)
    CXX = g++
endif
ifeq ($(CXX), icpc)
    CXXFLAGS = -mkl=sequential -O3 -fPIC -D NOMATLAB=1
else
    CXXFLAGS = -O3 -fPIC -D NOMATLAB=1 $(INCLUDES) $(LDFLAGS)
endif

# Recipes
.PHONY: all
all: clean $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(SRCS:.cpp=.d) : %.d : %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

.PHONY : clean
clean:
	rm -f $(TARGET) $(OBJS) $(SRCS:.cpp=.d)

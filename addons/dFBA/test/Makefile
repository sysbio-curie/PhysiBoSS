CC := g++

# Allow override from environment
ifdef PHYSICELL_CPP
  CC := $(PHYSICELL_CPP)
endif

ARCH := native

# For debugging with gdb just add: -g3 -ggdb -O3
CFLAGS      := -march=$(ARCH) -O3 -fomit-frame-pointer -mfpmath=both -fopenmp -m64 -std=c++11

# Dependencies
DEPS_DIR    := ../ext
HIGHS_DIR   := $(DEPS_DIR)/hiGHS
DEPS_CFLAGS := -I$(DEPS_DIR)/libsbml/include -I$(HIGHS_DIR)/src -I$(HIGHS_DIR)/build -I../src
LD_PATH     := -L$(DEPS_DIR)/libsbml/lib -L$(HIGHS_DIR)/build/lib

# Runtime rpath (so you don’t need LD_LIBRARY_PATH)
RPATH       := -Wl,-rpath,$(DEPS_DIR)/libsbml/lib -Wl,-rpath,$(HIGHS_DIR)/build/lib

## "linux-x64", "linux-x86", "win64", "win32", "osx"
PKG_ARCH := linux-x64

# --- Link against HiGHS ---
LIBS        := -lhighs -lsbml-static -lxml2 -lbz2 -lz

ifeq ($(OS),Windows_NT)
  WIN_LD_FLAG := -LC:\msys64\mingw64\bin
  ARCH := win64
else
  WIN_LD_FLAG :=
  UNAME_S := $(shell uname -s)
  ifeq ($(UNAME_S),Darwin)
    UNAME_P := $(shell uname -p)
    var := $(shell which $(CC) | xargs file)
    ifeq ($(lastword $(var)),arm64)
      CFLAGS := -march=$(ARCH) -O3 -fomit-frame-pointer -fopenmp -m64 -std=c++11
    endif
  endif
endif

COMPILE_COMMAND := $(CC) $(CFLAGS) $(DEPS_CFLAGS)

SRC_DIR := "./"

vpath %.o     $(SRC_DIR)
vpath %.cpp   $(SRC_DIR)
vpath %.h     $(SRC_DIR)

# Objects from ../src
ALL_OBJECTS := dfba_Metabolite.o dfba_Reaction.o dfba_Solution.o dfba_Model.o

# Use a concrete header as the sentinel that deps are present
LIBFBA_SENTINEL := $(HIGHS_DIR)/src/Highs.h

.PHONY: all make_lib clean libFBA test_sbml test_clp test_fba

make_lib: libFBA $(ALL_OBJECTS)
all: make_lib test_sbml test_clp test_fba

# --- Tests (link stage) ---
test_sbml: $(LIBFBA_SENTINEL) $(ALL_OBJECTS) ./main_test_sbml.cpp
	$(COMPILE_COMMAND) -o $@ $(ALL_OBJECTS) main_test_sbml.cpp $(LD_PATH) $(WIN_LD_FLAG) $(RPATH) $(LIBS)
	@echo created $@
	@echo

test_clp: $(LIBFBA_SENTINEL) $(ALL_OBJECTS) ./main_test_clp.cpp
	$(COMPILE_COMMAND) -o $@ $(ALL_OBJECTS) main_test_clp.cpp $(LD_PATH) $(WIN_LD_FLAG) $(RPATH) $(LIBS)
	@echo created $@
	@echo

test_fba: $(LIBFBA_SENTINEL) $(ALL_OBJECTS) ./main_test_fba.cpp
	$(COMPILE_COMMAND) -o $@ $(ALL_OBJECTS) main_test_fba.cpp $(LD_PATH) $(WIN_LD_FLAG) $(RPATH) $(LIBS)
	@echo created $@
	@echo

# --- Build dependencies once (creates headers and libs under ../ext) ---
libFBA: $(LIBFBA_SENTINEL)

$(LIBFBA_SENTINEL):
	python3 ../../../beta/setup_fba.py --pkg libsbml --arch $(PKG_ARCH)
	@echo "HiGHS library expected at $(HIGHS_DIR)/build/lib"

# --- Object files (compile stage; no link libs here) ---
dfba_intracellular.o: ../src/dfba_intracellular.cpp
	$(COMPILE_COMMAND) -c $< -o $@

dfba_Model.o: ../src/dfba_Model.cpp
	$(COMPILE_COMMAND) -c $< -o $@

dfba_Reaction.o: ../src/dfba_Reaction.cpp
	$(COMPILE_COMMAND) -c $< -o $@

dfba_Metabolite.o: ../src/dfba_Metabolite.cpp
	$(COMPILE_COMMAND) -c $< -o $@

dfba_Solution.o: ../src/dfba_Solution.cpp
	$(COMPILE_COMMAND) -c $< -o $@

clean:
	rm -f *.o
	rm -f ./test_*

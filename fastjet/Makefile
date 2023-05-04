# Makefile for fastjet timing executable
# Adjust these options for your system
HEPMC3_DIR=/opt
FASTJET_DIR=/opt

.PHONY: all clean

CXXFLAGS+=-MMD
CXXFLAGS+=-O2 -ggdb
CXXFLAGS+=-ggdb #For perf profiler
#CXXFLAGS+=-ggdb -Og

all: chep-polyglot-jets

clean:
	$(RM) chep-polyglot-jets *.o *.d

SRC = chep-polyglot-jets.cc

DEPS:=$(addsuffix .d,$(basename $(SRC)))

chep-polyglot-jets: chep-polyglot-jets.cc
	$(LINK.cc) -o $@ -Wno-deprecated-declarations  -I $(FASTJET_DIR)/include -L $(FASTJET_DIR)/lib -Wl,--rpath $(FASTJET_DIR)/lib \
-I $(HEPMC3_DIR)/include -L $(HEPMC3_DIR)/lib -Wl,--rpath $(HEPMC3_DIR) $< -l fastjet -lHepMC3 -lm


-include $(DEPS)


# Makefile for fastjet timing executable
# Adjust these options for your system, in particulat to point to
# the FastJet and HepMC3 install areas
HEPMC3_DIR=/opt
FASTJET_DIR=/opt

.PHONY: all clean

CXXFLAGS+=-MMD
CXXFLAGS+=-O2 -ggdb --std=c++17

all: chep-polyglot-jets

clean:
	$(RM) chep-polyglot-jets *.o *.d

SRC = chep-polyglot-jets.cc

DEPS:=$(addsuffix .d,$(basename $(SRC)))

chep-polyglot-jets: chep-polyglot-jets.cc
	$(LINK.cc) -o $@ -Wno-deprecated-declarations  -I $(FASTJET_DIR)/include -L $(FASTJET_DIR)/lib \
-I $(HEPMC3_DIR)/include -L $(HEPMC3_DIR)/lib $< -l fastjet -lHepMC3 -lm

-include $(DEPS)

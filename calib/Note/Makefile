.PHONY: all
order:=$(shell seq 5 5 30)
all: $(order:%=peTimeCoeff_Order%.pdf)

peTimeCoeff_Order%.pdf:
	python3 visual_coeff.py $(notdir $*) -o $(@) >$@.log 2>&1
.DELETE_ON_ERROR:
# Keep intermediate files around
.SECONDARY:
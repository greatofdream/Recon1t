.PHONY: all checkpe
order:=$(shell seq 5 5 30)
E:=339MeV
path:=/srv/greatofdream/simnew/newK+/zaxis/E_$(E)/
#radius:=$(shell seq  16000 -1000 7000) $(shell seq  -5000 -1000 -16000)
radius:=$(shell seq  16000 -1000 -16000)
peFiles:=$(radius:%=../coeff_pe_1t_339MeV/file_%.h5)
timeFiles:=$(radius:%=../coeff_time_1t_339MeV/file_%.h5)
geometry:=/srv/greatofdream/geometry/PMTPos_Acrylic_with_chimney.csv
all: $(order:%=peTimeCoeff_Order%.pdf)

peTimeCoeff_Order%.pdf: $(peFiles) $(timeFiles)
	python3 visual_coeff.py $(notdir $*) -p $(peFiles) -r $(radius) -t $(timeFiles) -o $(@) >$@.log 2>&1
checkpe.pdf: $(radius:%=$(path)%/wave.h5)
	python3 checkPe.py -f $^ -r $(radius) -o $(@) >$@.log 2>&1
checkgeo.pdf: $(geometry)
	python3 checkgeometry.py -g $^ -o $(@) >$@.log 2>&1
.DELETE_ON_ERROR:
# Keep intermediate files around
.SECONDARY:
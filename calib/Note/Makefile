.PHONY: all checkpez
order:=$(shell seq 5 5 30)
E:=339MeV
particle:=e-
path:=/junofs/users/junoprotondecay/zhangaq/simnew/$(particle)/zaxis/E_339MeV
radius:=$(shell seq  17000 -1000 -17000)
peFiles:=$(radius:%=../coeff_pe_1t_339MeV/file_%.h5)
timeFiles:=$(radius:%=../coeff_time_1t_339MeV/file_%.h5)
geometry:=/srv/greatofdream/geometry/PMTPos_Acrylic_with_chimney.csv
all: $(order:%=peTimeCoeff_Order%.pdf)
checkpez: $(radius:%=check/$(particle)/checkpez%.pdf)

peTimeCoeff_Order%.pdf: $(peFiles) $(timeFiles)
	python3 visual_coeff.py $(notdir $*) -p $(peFiles) -r $(radius) -t $(timeFiles) -o $(@) >$@.log 2>&1
checkpe.pdf: $(radius:%=$(path)/%/wave.h5)
	python3 checkPe.py -f $^ -r $(radius) -o $(@) >$@.log 2>&1
check/$(particle)/checkpez%.pdf: $(path)/%/wave.h5
	mkdir -p $(dir $@)
	python3 checkPez.py -f $^ -r $(notdir $*) -g $(geometry) -o $(@) >$@.log 2>&1
PE.pdf: $(radius:%=$(path)/%/wave.h5)
	./checkPe.R -o $(@) -- -i $^
checkgeo.pdf: $(geometry)
	python3 checkgeometry.py -g $^ -o $(@) >$@.log 2>&1
.DELETE_ON_ERROR:
# Keep intermediate files around
.SECONDARY:

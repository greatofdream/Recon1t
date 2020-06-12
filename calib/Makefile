.PHONY: all
export OMP_NUM_THREADS=2
E:=339MeV
particle:=e-
#path:=/srv/greatofdream/simnew/newK+/zaxis/E_$(E)/
path:=/srv/greatofdream/simnew/$(particle)/zaxis/E_$(E)/
upper:=+17000
lower:=-17000
radius:=$(shell seq  $(upper) -1000 $(lower))
save_pe:=coeff_pe_1t_$(E)
save_time:=coeff_time_1t_$(E)
src_pe:=$(radius:%=$(save_pe)/file_%.h5)
src_time:=$(radius:%=$(save_time)/file_%.h5)
geometry:=/srv/greatofdream/geometry/PMTPos_Acrylic_with_chimney.csv

all:coeff_pe coeff_time

$(save_pe)/file_%.h5:
	mkdir -p $(save_pe)
	env TIME="time result\ncmd:%C\nreal %es\nuser %Us \nsys  %Ss \nmemory:%MKB \ncpu %P" time python3 PE_calib/main_calib.py $* $(path) $@ $(geometry) >$@.log 2>&1

$(save_time)/file_%.h5:
	mkdir -p $(save_time)
	env TIME="time result\ncmd:%C\nreal %es\nuser %Us \nsys  %Ss \nmemory:%MKB \ncpu %P" time python3 Time_calib/main_calib.py $* $(path) $@ $(geometry) >$@.log 2>&1

coeff_pe:$(src_pe)
	python3 PE_calib/Gather.py ./$(save_pe)/ $(upper) $(lower) 35 >$@.log 2>&1

coeff_time:$(src_time)
	python3 Time_calib/Gather.py ./$(save_time)/ $(upper) $(lower) 35 >$@.log 2>&1

# Delete partial files when the processes are killed.
.DELETE_ON_ERROR:
# Keep intermediate files around
.SECONDARY:

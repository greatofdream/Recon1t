.PHONY: all

E:=2.0MeV_dn
path:=/mnt/stage/douwei/Simulation/1t_root/$(E)/

upper_i:=0.40
lower_i:=0.00
radius_i:=$(shell seq -f '%.3f' $(upper_i) -0.01 $(lower_i))

upper_o:=0.65
lower_o:=0.40
radius_o:=$(shell seq -f '%.3f' $(upper_o) -0.002 $(lower_o))

radius:=$(shell echo $(radius_i) $(radius_o))

save_pe:=coeff_pe_1t_$(E)_Lasso
save_time:=coeff_time_1t_$(E)_Lasso

src_pe:=$(radius:%=$(save_pe)/file_%.h5)
src_time:=$(radius:%=$(save_time)/file_%.h5)

order:=25

all:coeff_pe coeff_time
#all:$(src_pe) $(src_time)
$(save_pe)/file_%.h5:
	mkdir -p $(save_pe)
	python3 PE_calib/main_calib.py $* $(path) $@ $(order) > logs/PE$*.log

$(save_time)/file_%.h5:
	mkdir -p $(save_time)
	python3 Time_calib/main_calib.py $* $(path) $@ $(order) > logs/Time$*.log

coeff_pe:$(src_pe)
	python3 PE_calib/Gather.py ./$(save_pe)/ $(upper_o) $(lower_o) $(lower_i) $(order)

coeff_time:$(src_time)
	python3 Time_calib/Gather.py ./$(save_time)/ $(upper_o) $(lower_o) $(lower_i) $(order)

# Delete partial files when the processes are killed.
.DELETE_ON_ERROR:
# Keep intermediate files around
.SECONDARY:

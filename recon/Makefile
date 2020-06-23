.PHONY: all
export OMP_NUM_THREADS=2
particle:=e-
order:=5
E:=2.0MeV
method:=sphereFitallTime
radius:=$(shell seq  17000 -1000 -17000)
timelimit:=1000
# path:=/junofs/users/junoprotondecay/zhangaq/simnew/newK+/zaxis/E_339MeV
path:=/junofs/users/junoprotondecay/zhangaq/e/e-/zaxis/E_339MeV
save:=/srv/greatofdream/data/$(particle)/recon
# sim:=$(wildcard $(path)/*/detsim_user.root)
sim:=$(radius:%=$(path)/%/detsim_user.root)
dst:=$(sim:$(path)/%/detsim_user.root=$(save)/%/$(method)/recon.h5)
# geometry:=/srv/greatofdream/geometry/PMTPos_Acrylic_with_chimney.csv
geometry:=/cvmfs/juno.ihep.ac.cn/sl6_amd64_gcc830/Pre-Release/J20v1r0-Pre2/offline/Simulation/DetSimV2/DetSimOptions/data/PMTPos_Acrylic_with_chimney.csv
peOrderFile:=../calib/coeff_pe_339MeV_det$(timelimit)/coeffpe_fit.h5
timeOrderFile:=../calib/coeff_time_339MeV_det$(timelimit)/coefftime_fit.h5

all:$(dst)
det: $(sim:$(path)/%/detsim_user.root=$(save)det$(timelimit)/%/$(method)/recon.h5)
$(save)/%/$(method)/recon.h5: $(path)/%/det_user.root $(path)/%/wave.root
	mkdir -p $(dir $@)
	python3 Recon.py -t $< -e $(word 2,$^) -d $(order) -g $(geometry) -o $@> $@.log 2>&1

$(save)det$(timelimit)/%/$(method)/recon.h5: $(path)/%/det_user.root
	mkdir -p $(dir $@)
	python3 reconDet.py -t $< -T $(timelimit) -O $(order) -d $(peOrderFile) $(timeOrderFile) -g $(geometry) -o $@> $@.log 2>&1
error: $(particle)/$(method)/error.pdf
$(particle)/$(method)/error.pdf: $(wildcard $(save)det$(timelimit)/*/$(method)/recon.h5)
	mkdir -p $(dir $@)
	python3 visual_result.py $^ -p $(^:$(save)det$(timelimit)/%/$(method)/recon.h5=$(path)/%/det_user.root) -z $(^:$(save)det$(timelimit)/%/$(method)/recon.h5=%) -o $@ >$@.log 2>&1
# Delete partial files when the processes are killed.
.DELETE_ON_ERROR:
# Keep intermediate files around
.SECONDARY:

.PHONY: all
particle:=e-
order:=10
E:=2.0MeV
method:=sphere
radius:=$(shell seq  16000 -1000 -16000)
# path:=/junofs/users/junoprotondecay/zhangaq/simnew/newK+/zaxis/E_339MeV
path:=/junofs/users/junoprotondecay/zhangaq/e/e-/zaxis/E_339MeV
save:=/srv/greatofdream/data/$(particle)/recon
# sim:=$(wildcard $(path)/*/detsim_user.root)
sim:=$(radius:%=$(path)/%/detsim_user.root)
dst:=$(sim:$(path)/%/detsim_user.root=$(save)/%/$(method)/recon.h5)
geometry:=/srv/greatofdream/geometry/PMTPos_Acrylic_with_chimney.csv

all:$(dst)

$(save)/%/$(method)/recon.h5: $(path)/%/det_user.root $(path)/%/wave.root
	mkdir -p $(dir $@)
	python3 Recon.py -t $< -e $(word 2,$^) -d $(order) -g $(geometry) -o $@> $@.log
error.pdf: $(wildcard $(save)/*/$(method)/recon.h5)
	python3 note/visual_result.py $^ -p $(^:$(save)/%/$(method)/recon.h5=$(path)/%/detsim_user.root) -z $(^:$(save)/%/$(method)/recon.h5=%) -o $@
# Delete partial files when the processes are killed.
.DELETE_ON_ERROR:
# Keep intermediate files around
.SECONDARY:

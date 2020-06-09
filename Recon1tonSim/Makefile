.PHONY: all

order:=10
E:=2.0MeV
method:=sphere
path:=/home/greatofdream/protondecay/src/data/simulate/truth/
save:=/home/greatofdream/protondecay/src/data/recon/
sim:=$(wildcard $(path)*/detsim_user.root)
dst:=$(sim:$(path)%/detsim_user.root=$(save)%/$(method)/recon.h5)
geometry:=/home/greatofdream/protondecay/src/data/geometry/PMTPos_Acrylic_with_chimney.csv

all:$(dst)

$(save)%/$(method)/recon.h5:$(path)%/detsim_user.root
	mkdir -p $(dir $@)
	python3 Recon.py $^ $@ $(order) $(geometry)> $@.log

# Delete partial files when the processes are killed.
.DELETE_ON_ERROR:
# Keep intermediate files around
.SECONDARY:

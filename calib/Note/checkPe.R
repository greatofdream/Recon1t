#!/usr/bin/env Rscript
require(argparser)
require(plyr)
require(doMC)
registerDoMC()
require(rhdf5)
require(data.table)
require(ggplot2)

psr <- arg_parser("event plot")
psr <- add_argument(psr, "--input", nargs=Inf, help="input files, wave.h5")
psr <- add_argument(psr, "-o", help="out files")

argv <- parse_args(psr)

loadf <- function(fn) {
    fid <- H5Fopen(fn)
    rst <- cbind(h5read(fid, "GroundTruth"), z=h5readAttributes(fid, "/")$z)
    H5Fclose(fid)
    rst
}

d <- data.table(ldply(argv$input, loadf, .parallel=TRUE))

PE <- d[, .N, by=c("EventID", "z")]

pdf(argv$o, 16, 7)
PE$z <- as.factor(PE$z)

p <- ggplot(PE, aes(z, N)) + geom_boxplot() + xlab("z/mm") + ylab("Number of PEs")
print(p)

evn <- PE[, .N, by=z]
p <- ggplot(evn, aes(z, N)) + geom_bar(stat="identity") + xlab("z/mm") + ylab("Number of Events")
print(p)
dev.off()

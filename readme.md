
Monorepo for the obelisk project

**axon** and **emergent** are submodules *for now*. After you clone you must do:

`git submodule init`

`git submodule update`



### Benchmark Protobrain
This is the standard test that you can judge your changes by.
TODO


### Profiling Protobrain
1) Set `PROFILE=true` in brain.cfg
2) run for some time 
3) > `go tool pprof cpu.pprof`


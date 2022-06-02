
Monorepo for the obelisk project

**axon** and **emergent** are submodules *for now*. After you clone you must do:

`git submodule init`

`git submodule update`

## Submitting Changes

Submit a pull request rebased on top of master

 * Include a descriptive commit message.
 * Changes contributed via pull request should focus on a single issue at a time.


## Stellar Core Contribution Specifics

### General
* Try to separate logically distinct changes into separate commits and thematically distinct
  commits into separate pull requests.
* Please ensure that all tests pass before submitting changes. 

### Keeping our commit history clean

We're striving to keep master's history with minimal merge bubbles. To achieve this, we're asking
PRs to be submitted rebased on top of master.

To keep your local repository in a "rebased" state, simply run:
* `git config branch.autosetuprebase always` _changes the default for all future branches_
* `git config branch.master.rebase true` _changes the setting for branch master_

Note: you may still have to run manual "rebase" commands on your branches, to rebase on top of
master as you pull changes from upstream.

### Testing

Please ensure that all tests pass before submitting changes. 



### Benchmark Protobrain
This is the standard test that you can judge your changes by.
TODO


### Profiling Protobrain
1) Set `PROFILE=true` in brain.cfg
2) run for some time 
3) > `go tool pprof cpu.pprof`


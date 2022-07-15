#! /bin/bash
# This script sets up the project beyond pulling the repo
# used by ODPW

git submodule update --init --recursive
#cd worlds/network
#thrift -r --gen go:package_prefix=github.com/Astera-org/obelisk/worlds/network/gengo/ -out gengo/ env.thrift
cd worlds
go mod tidy
cd ../models
go mod tidy
cd ..



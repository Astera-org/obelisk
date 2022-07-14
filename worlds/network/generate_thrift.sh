# we don't use the default gen-py name because python hates hyphens on packages
mkdir -p genpy
mkdir -p gengo
mkdir -p gencpp

thrift -r --gen py -out genpy/ env.thrift
thrift -r --gen go:package_prefix=github.com/Astera-org/obelisk/worlds/network/gengo/ -out gengo/ env.thrift
thrift -r --gen cpp -out gencpp/ env.thrift

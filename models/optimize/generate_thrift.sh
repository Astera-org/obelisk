# we don't use the default gen-py name because python hates hyphens on packages
mkdir -p genpy
mkdir -p gengo
#mkdir -p gencpp

thrift -r --gen py -out genpy/ optimize.thrift
thrift -r --gen go -out gengo/ optimize.thrift
#thrift -r --gen cpp -out gencpp/ optimize.thrift

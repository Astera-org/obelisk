# we don't use the default gen-py name because python hates hyphens on packages
#mkdir -p genpy
mkdir -p gengo
#mkdir -p gencpp

#thrift -r --gen py -out genpy/ infra.thrift
thrift -r --gen go -out gengo/ infra.thrift
#thrift -r --gen cpp -out gencpp/ infra.thrift

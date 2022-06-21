mkdir -p gengo
mkdir -p genjs

thrift -r --gen go -out gengo/ infra.thrift
thrift -r --gen js -out genjs/ infra.thrift

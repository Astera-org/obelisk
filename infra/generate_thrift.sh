mkdir -p gengo
mkdir -p genjs

thrift -r --gen go:package_prefix=github.com/Astera-org/obelisk/infra/gengo/ -out gengo/ infra.thrift
thrift -r --gen js -out genjs/ infra.thrift

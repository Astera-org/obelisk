cd ../models/integrated/protobrain
go build
cd ../../../run
copy ..\models\integrated\protobrain\protobrain.exe protobrain.exe


cd ../worlds/integrated/fworld
go build
cd ../../../run
copy ..\worlds\integrated\fworld\fworld.exe fworld.exe

cd ../infra/mock
go build
cd ../../run
copy ..\infra\mock\mock.exe mock.exe


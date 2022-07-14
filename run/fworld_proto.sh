# starts fworld and protobrain

pushd .
cd ../models/integrated/protobrain
go run *.go &

sleep 5

popd
pushd .
cd ../worlds/integrated/fworld
go run *.go

popd

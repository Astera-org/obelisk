# starts egan and protobrain

pushd .
cd ../models/integrated/protobrain
go run . &

sleep 8

popd
pushd .
cd ../worlds/integrated/fworld
go run .

popd

# Run protobrain and fworld, such that they connect to each other

# Run this from obelisk/

go run $(ls models/integrated/protobrain/*.go | grep -v _test) &

sleep 5

go run $(ls worlds/integrated/fworld/*.go | grep -v _test) &

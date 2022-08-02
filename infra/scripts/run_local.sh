# Helper script to help you run all services locally

# What services do we need?
# jobczar, worker, binserver, odpw
# vasco? optimizer?

s_flag=false

print_usage() {
  echo "This script starts or stops all infra services locally"
  echo "Usage: $0 [-s]"
  echo "  -s      kills all services and exits"
}

while getopts 's' flag; do
  case "${flag}" in
    s) s_flag=true ;;
    *) print_usage
       exit 1 ;;
  esac
done

services=(jobczar binserver odpw worker)

if [ $s_flag == true ] ; then
  echo "Stop all"
  for s in ${services[@]}
  do
    echo "Stopping $s"
    # slash to avoid matching other processes (for example kworker)
    pkill -f "/$s"
  done

  exit
fi

pushd .
cd ..

for s in ${services[@]}
do
  echo "Starting $s"
  # slash to avoid matching other processes (for example kworker)
  pkill -f "/$s"
  cd $s
  go run . 1>> ../scripts/start.log 2>> ../scripts/error.log &
  # without this the worker sometimes fails to connect to jobczar
  # that's probably a bug in thrift because it should attempt to reconnect
  sleep 5
  cd ..
done

popd

# TODO: tail all service logs
tail -f  error.log start.log ../worker/worker.log ../jobczar/jobczar.log \
    ../binserver/binserver.log ../odpw/odpw.log

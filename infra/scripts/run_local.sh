# Helper script to help you run all services locally

# What services do we need?
# jobczar, worker, binserver, odpw
# vasco? optimizer?

s_flag=false

# TODO
print_usage() {
  printf "Usage: ..."
}

while getopts 's' flag; do
  case "${flag}" in
    s) s_flag=true ;;
    *) print_usage
       exit 1 ;;
  esac
done

services=(jobczar worker)

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

cd ..

for s in ${services[@]}
do
  echo "Starting $s"
  # slash to avoid matching other processes (for example kworker)
  pkill -f "/$s"
  cd $s
  go run . > /dev/null 2>&1 &
  # without this the worker sometimes fails to connect to jobczar
  # that's probably a bug in thrift because it should attempt to reconnect
  sleep 5
  cd ..
done

# TODO: tail all service logs
tail -f  worker/worker.log jobczar/jobczar.log

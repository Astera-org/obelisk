cd ../infra/jobczar
go build
cd ../../run
copy ..\infra\jobczar\jobczar.exe jobczar.exe


cd ../infra/worker
go build
cd ../../run
copy ..\infra\worker\worker.exe worker.exe


cd ../infra/binserver
go build
cd ../../run
copy ..\infra\binserver\binserver.exe binserver.exe


Folder to contain our infrasructure projects: https://docs.google.com/document/d/1Lacr8biGy1chESkKfiCTrKx6BiBofO2ocWYd8lQdABU/edit#



## Building

### Compiling thrift files
> thrift -r --gen go -out gengo/ infra.thrift


## Setup

Jobboard is the ui and it fetches data from Jobczar.

Jobczar uses mysql to store the data. Make sure you have mysql installed and running on your machine.
Then, run this to setup the database and tables. This also adds some mock data to help in development.
Careful running this script drops the entire database every time.

If your mysql installation does not have a password or has a different authnetication scheme remove the -p flag which asks for a password.

### mysql
> mysql -u username -p < database.sql

package main

import (
	"database/sql"
	"fmt"

	log "github.com/Astera-org/easylog"

	_ "github.com/go-sql-driver/mysql"
)

/*
type JobInfo struct {
	jobID int32
	workerName string
}*/

type Database struct {
	db *sql.DB
}

func (db *Database) Connect() {
	var err error
	db.db, err = sql.Open("mysql", gConfig.DB_CONNECT)
	if err != nil {
		panic(err)
	}
}

func (db *Database) getNameVersion(binID int) (string, string) {
	var name, version string
	sql := fmt.Sprint("SELECT name, version FROM binaries where bin_id = ", binID)
	err := db.db.QueryRow(sql).Scan(&name, &version)
	if err != nil {
		log.Error(err)
		return "", ""
	}
	return name, version
}

func (db *Database) getJobInfo(jobID int) (int, string, error) {
	var status int
	var workerName string
	sql := fmt.Sprint("SELECT status,worker_name FROM jobs where job_id = ", jobID)
	err := db.db.QueryRow(sql).Scan(&status, &workerName)
	if err != nil {
		log.Error(err)
		return -1, "", err
	}
	return status, workerName, nil
}

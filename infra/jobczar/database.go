package main

import (
	"fmt"
	log "github.com/Astera-org/easylog"
	"github.com/Astera-org/obelisk/infra/gengo/infra"
	_ "github.com/go-sql-driver/mysql"
	"github.com/jmoiron/sqlx"
)

type Database struct {
	db *sqlx.DB
}

func (db *Database) Connect() {
	var err error
	db.db, err = sqlx.Open("mysql", gConfig.DB_CONNECT)
	if err != nil {
		panic(err)
	}
}

func (db *Database) GetJobCount(status int32) int32 {
	var count int32 = -1
	err := db.db.Get(&count, "SELECT COUNT(*) FROM jobs WHERE status = ?", status)
	if err != nil {
		log.Error(err)
	}
	return count
}

func (db *Database) GetBinInfo(binID int32) *infra.BinInfo {
	binInfo := infra.BinInfo{}
	err := db.db.Get(&binInfo, "SELECT * FROM binaries where bin_id = ?", binID)
	if err != nil {
		log.Error(err)
		return nil
	}
	return &binInfo
}

func (db *Database) QueryJobs() ([]*infra.JobInfo, error) {
	query := "SELECT * from jobs order by job_id desc LIMIT 1000"
	res := []*infra.JobInfo{}
	err := gDatabase.db.Select(&res, query)
	if err != nil {
		log.Error(err)
		return nil, err
	}
	return res, nil
}

func (db *Database) GetBinInfos(filterBy string) ([]*infra.BinInfo, error) {
	query := "SELECT * FROM binaries order by time_added desc"

	if filterBy != "" {
		query = fmt.Sprintf(
			`SELECT * FROM binaries where %s order by time_added desc`, filterBy)
	}

	res := []*infra.BinInfo{}
	err := gDatabase.db.Select(&res, query)
	if err != nil {
		log.Error(err)
		return nil, err
	}
	return res, nil
}

func (db *Database) GetCallback(jobID int32) string {
	var callback string = ""
	err := db.db.Get(&callback, "SELECT callback from jobs where job_id = ?", jobID)
	if err != nil {
		log.Error(err)
	}
	return callback
}

package main

import (
	"database/sql"
	"fmt"

	log "github.com/Astera-org/easylog"
	"github.com/Astera-org/obelisk/infra/gengo/infra"
	_ "github.com/go-sql-driver/mysql"
)

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

func (db *Database) GetJobCount(status int32) int32 {
	var count int32
	sql := fmt.Sprintf("SELECT COUNT(*) FROM jobs WHERE status = %d", status)
	err := db.db.QueryRow(sql).Scan(&count)
	if err != nil {
		log.Error(err)
	}
	return count
}

func (db *Database) GetBinInfo(binID int32) *infra.BinInfo {
	sql := fmt.Sprintf("SELECT name, version,package_hash FROM binaries where bin_id = %d", binID)
	row := db.db.QueryRow(sql)
	binInfo := infra.BinInfo{}
	err := row.Scan(&binInfo.Name, &binInfo.Version, &binInfo.PackageHash)
	if err != nil {
		log.Error(err)
		return nil
	}
	return &binInfo
}

func (db *Database) GetBinInfos(filterBy string) ([]*infra.BinInfo, error) {
	sql := fmt.Sprintf(
		`SELECT bin_id, name, version, package_hash, time_added, type, status
                FROM binaries order by time_added desc`)

	if filterBy != "" {
		sql = fmt.Sprintf(
			`SELECT bin_id, name, version, package_hash, time_added, type, status
                FROM binaries where %s order by time_added desc`, filterBy)
	}

	rows, err := gDatabase.db.Query(sql)
	if err != nil {
		log.Error(err)
		return nil, err
	}

	res := make([]*infra.BinInfo, 0)
	for rows.Next() {
		bi := infra.BinInfo{}
		err := rows.Scan(&bi.BinID, &bi.Name, &bi.Version, &bi.PackageHash, &bi.TimeAdded, &bi.Type, &bi.Status)
		if err == nil {
			res = append(res, &bi)
		} else {
			log.Error(err)
		}
	}
	return res, nil
}

func (db *Database) GetCallback(jobID int32) string {
	var callback string = ""
	sql := fmt.Sprint("SELECT callback from jobs where job_id =", jobID)
	err := db.db.QueryRow(sql).Scan(&callback)
	if err != nil {
		log.Error(err)
	}
	return callback
}

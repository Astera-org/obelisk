package main

import (
	"database/sql"
	"fmt"

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
		fmt.Println(err)
	}
	return count
}

func (db *Database) GetBinInfo(binID int32) *infra.BinInfo {
	sql := fmt.Sprintf("SELECT name, version,hash FROM binaries where bin_id = %d", binID)
	row := db.db.QueryRow(sql)
	binInfo := infra.BinInfo{}
	err := row.Scan(&binInfo.Name, &binInfo.Version, &binInfo.Hash)
	if err != nil {
		fmt.Println(err)
		return nil
	}
	return &binInfo
}

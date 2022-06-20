package main

import (
	"database/sql"
	"fmt"

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

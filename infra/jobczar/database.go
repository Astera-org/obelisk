package main

import (
	"database/sql"

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

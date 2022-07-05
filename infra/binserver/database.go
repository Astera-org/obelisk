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

func (db *Database) getNameVersion(binID int) (string, string) {
	var name, version string
	sql := fmt.Sprint("SELECT name, version FROM binaries where id = ", binID)
	err := db.db.QueryRow(sql).Scan(&name, &version)
	if err != nil {
		fmt.Println(err)
		return "", ""
	}
	return name, version
}

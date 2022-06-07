package main

import "database/sql"

type Database struct {
	db *sql.DB
}

func (db *Database) Connect() {
	var err error
	db.db, err = sql.Open("postgres", gConfig.DB_CONNECT)
	if err != nil {
		panic(err)
	}
}

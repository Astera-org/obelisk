package main
import "database/sql"

type Database struct {
	db DB
}

func (db *Database)Connect() {
	db.db, err := sql.Open("postgres", gConfig.DB_CONNECT)
	if err != nil {
  		panic(err)
	}
}
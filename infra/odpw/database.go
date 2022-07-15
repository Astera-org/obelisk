package main

import (
	"database/sql"
	"fmt"

	log "github.com/Astera-org/easylog"
	_ "github.com/go-sql-driver/mysql"
)

type Database struct {
	db *sql.DB
}

func (db *Database) Connect() {
	var err error
	db.db, err = sql.Open("mysql", gConfig.DB_CONNECT)
	if err != nil {
		log.Fatal(err)
	}
}

func (db *Database) DoesExist(hash string) bool {
	var count int
	sql := "SELECT COUNT(*) FROM binaries WHERE package_hash = \"" + hash + "\""
	err := db.db.QueryRow(sql).Scan(&count)
	if err != nil {
		log.Error(err)
		return true
	}
	return count > 0
}

// name VARCHAR(10), version VARCHAR(10),
// hash VARCHAR(40), time_added TIMESTAMP DEFAULT NOW(), type INT DEFAULt 0, status INT DEFAULT 0
func (db *Database) AddBinary(name string, version string, packageHash string, binType int) (int, error) {
	sql := fmt.Sprintf("INSERT INTO binaries (name, version,package_hash,type,status) VALUES ('%s','%s','%s',%d,2)",
		name, version, packageHash, binType)

	result, err := db.db.Exec(sql)
	if err != nil {
		log.Error(err)
		return -1, err
	}
	insertID, err := result.LastInsertId()
	if err != nil {
		log.Error(err)
		return -1, err
	}

	return int(insertID), nil
}

// Should we actually connect to JobCzar to do this?
//     Only if we don't want to have the DB accessable to odpw
func (db *Database) AddJob(userID int, agentID int, worldID int) (int32, error) {
	sql := fmt.Sprintf("INSERT into jobs (user_id,priority,callback,agent_id,world_id,agent_param,world_param,note) values (%d,10,'%s',%d,%d,'','','regression test')",
		userID, gConfig.CALLBACK_URL, agentID, worldID)

	result, err := db.db.Exec(sql)
	if err != nil {
		return -1, err
	}
	insertID, err := result.LastInsertId()
	if err != nil {
		return -1, err
	}
	return int32(insertID), nil
}

func (db *Database) GetLatestID(binName string) (int, error) {
	var id int
	sql := fmt.Sprintf("SELECT bin_id FROM binaries WHERE name = '%s' and status=0 ORDER BY time_added DESC LIMIT 1", binName)
	err := db.db.QueryRow(sql).Scan(&id)
	if err != nil {
		log.Error(sql)
		log.Error(err)
		return 0, err
	}
	return id, nil
}

func (db *Database) UpdateVersion(oldVersion string, newVersion string) {
	sql := fmt.Sprintf("UPDATE binaries SET version = '%s' WHERE version = '%s'", newVersion, oldVersion)
	_, err := db.db.Exec(sql)
	if err != nil {
		log.Error(err)
	}
}

// set the status of the binary to status
func (db *Database) SetStatus(binID int, status int) {
	sql := fmt.Sprintf("UPDATE binaries SET status = %d WHERE bin_id = %d", status, binID)
	_, err := db.db.Exec(sql)
	if err != nil {
		log.Error(err)
	}
}

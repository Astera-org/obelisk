package main

import (
	"database/sql"
	"errors"
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

func (db *Database) QueryJobs(filterBy string) ([]*infra.JobInfo, error) {
	query := "SELECT * from jobs order by job_id desc LIMIT 1000"
	if filterBy != "" {
		query = fmt.Sprintf("SELECT * from jobs WHERE %s order by job_id desc LIMIT 1000", filterBy)
	}
	res := []*infra.JobInfo{}
	err := db.db.Select(&res, query)
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
	err := db.db.Select(&res, query)
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

func (db *Database) FetchWork(workerName, instanceName string) (*infra.Job, error) {
	job := infra.Job{}
	query := "SELECT * FROM jobs where status=0 order by priority desc LIMIT 1"
	err := db.db.Get(&job, query)
	if err == sql.ErrNoRows {
		log.Error(err)
		return &job, errors.New("empty")
	}
	if err != nil {
		log.Error(err)
		return &job, errors.New("db error")
	}

	sql := fmt.Sprintf("UPDATE jobs set status=1, worker_name='%s', instance_name='%s', time_handed=now() where job_id=%d",
		workerName, instanceName, job.JobID)
	_, err = db.db.Exec(sql)
	if err != nil {
		log.Error(err)
	}

	return &job, nil
}

func (db *Database) UpdateGoodJob(result *infra.ResultJob) (sql.Result, error) {
	sql := fmt.Sprintf("UPDATE jobs set status=2, seconds=%d, steps=%d,cycles=%d,score=%f where job_id=%d",
		result.Seconds, result.Steps, result.Cycles, result.Score, result.JobID)
	return db.db.Exec(sql)
}

func (db *Database) UpdateFailedJob(result *infra.ResultJob) (sql.Result, error) {
	sql := fmt.Sprintf("UPDATE jobs set status=0, worker_name='', instance_name='' where job_id=%d", result.JobID)
	return gDatabase.db.Exec(sql)
}

func (db *Database) AddJob(agentId int32, worldId int32,
	agentParam string, worldParam string, priority int32, userId int32, note string) (int64, error) {
	sql := fmt.Sprintf("INSERT into jobs (user_id,priority,agent_id,world_id,agent_param,world_param,note) values (%d,%d,%d,%d,'%s','%s','%s')",
		userId, priority, agentId, worldId, agentParam, worldParam, note)
	result, err := gDatabase.db.Exec(sql)
	if err != nil {
		log.Error(err)
		return -1, err
	}
	insertID, err := result.LastInsertId()
	if err != nil {
		log.Error(err)
		return -1, err
	}
	return insertID, nil
}

func (db *Database) RemoveJob(jobID int32) (bool, error) {
	sql := fmt.Sprintf("DELETE from jobs where job_id=%d and status=0", jobID)
	_, err := gDatabase.db.Exec(sql)
	if err != nil {
		return false, err
	}
	return true, nil
}

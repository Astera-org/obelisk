package main

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/Astera-org/obelisk/infra/gengo/infra"
)

type RequestHandler struct {
	infra.JobCzar
}

func (handler RequestHandler) FetchWork(ctx context.Context, workerName string, instanceName string) (*infra.Job, error) {

	job := infra.Job{}

	rows, err := gDatabase.db.Query("SELECT job_id,agent_name,world_name FROM jobs where status=0 order by priority desc LIMIT 1")
	if err != nil {
		fmt.Println(err)
		return &job, errors.New("db error")
	}

	err = rows.Scan(&job.JobID, &job.AgentName, &job.WorldName)
	if err != nil {
		fmt.Println(err)
		return &job, errors.New("empty")
	}

	sql := "UPDATE jobs set woker_name=$1, instance_name=$2 where job_id=$3"
	gDatabase.db.Exec(sql, workerName, instanceName, job.JobID)

	return &job, nil
}

func (handler RequestHandler) submitResult(ctx context.Context, result infra.ResultWork) (bool, error) {

	sql := "UPDATE jobs set cycles=$1,time_start=$2,timeStop=$3,score=$4 where job_id=$5"
	_, err := gDatabase.db.Exec(sql, result.Cycles, result.TimeStart, result.TimeStop, result.Score, result.JobID)
	if err != nil {
		fmt.Println(err)
		return false, err
	}

	return true, nil
}

func (handler RequestHandler) AddJob(ctx context.Context, agentName string, worldName string,
	agentCfg string, worldCfg string, priority int32, userID int32) (int32, error) {

	sql := "INSERT into jobs (user_id,priority,agent_name,world_name,agent_param,world_param) values ($1,$2,$3,$4,$5,$6)"
	_, err := gDatabase.db.Exec(sql, userID, priority, agentName, worldName, agentCfg, worldCfg)

	if err != nil {
		fmt.Println(err)
		return 0, err
	}

	rows, err := gDatabase.db.Query("SELECT LAST_INSERT_ID()")
	if err != nil {
		fmt.Println(err)
		return 0, errors.New("db error")
	}
	var insertID int32
	err = rows.Scan(&insertID)
	if err != nil {
		fmt.Println(err)
		return 0, errors.New("empty")
	}

	return insertID, nil
}

// only allow you to delete unservered up jobs
func (handler RequestHandler) RemoveJob(jobID int32) (bool, error) {
	sql := "DELETE from jobs where job_id=$1 and status=0"
	_, err := gDatabase.db.Exec(sql, jobID)
	if err != nil {
		return false, err
	}
	return true, nil
}

func (handler RequestHandler) RunSQL(ctx context.Context, sql string) (string, error) {
	rows, err := gDatabase.db.Query(sql)
	if err != nil {
		fmt.Println(err)
		return "error", errors.New("db error")
	}

	return printDBResult(rows), nil
}

func printDBResult(rows *sql.Rows) string {
	// Get column names
	columns, err := rows.Columns()
	if err != nil {
		return err.Error()
	}

	// Make a slice for the values
	values := make([]sql.RawBytes, len(columns))

	// rows.Scan wants '[]interface{}' as an argument, so we must copy the
	// references into such a slice
	// See http://code.google.com/p/go-wiki/wiki/InterfaceSlice for details
	scanArgs := make([]interface{}, len(values))
	for i := range values {
		scanArgs[i] = &values[i]
	}

	var retString string

	// Fetch rows
	for rows.Next() {
		// get RawBytes from data
		err = rows.Scan(scanArgs...)
		if err != nil {
			retString += err.Error()
		}

		// Now do something with the data.
		// Here we just print each column as a string.
		var value string
		for i, col := range values {
			// Here we can check if the value is nil (NULL value)
			if col == nil {
				value = "NULL"
			} else {
				value = string(col)
			}
			retString += fmt.Sprintf(columns[i], ": ", value)
		}
		retString += "-----------------------------------"
	}
	if err = rows.Err(); err != nil {
		retString += err.Error()
	}
	return retString
}

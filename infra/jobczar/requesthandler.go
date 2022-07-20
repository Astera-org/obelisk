package main

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	log "github.com/Astera-org/easylog"
	"github.com/Astera-org/obelisk/infra/gengo/infra"
)

type RequestHandler struct {
	infra.JobCzar
}

const (
	goodJob      int32 = 0
	jobFailed          = 1
	malformedJob       = 2
)

// TODO: move database guts into the database file
func (handler RequestHandler) FetchWork(ctx context.Context, workerName string, instanceName string) (*infra.Job, error) {
	job := infra.Job{}
	query := "SELECT job_id,agent_id,world_id,agent_param,world_param FROM jobs where status=0 order by priority desc LIMIT 1"
	err := gDatabase.db.Get(&job, query)
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
	_, err = gDatabase.db.Exec(sql)
	if err != nil {
		log.Error(err)
	}

	return &job, nil
}

// tell anyone that was waiting for this job to complete
func resultCallback(result *infra.ResultJob) {
	url := gDatabase.GetCallback(result.JobID)
	if url != "" {
		// post json of results to callback
		json, _ := json.MarshalIndent(result, "", " ")

		_, err := http.Post(url, "application/json", bytes.NewBuffer(json))
		if err != nil {
			log.Error(err)
		}
	}
}

func (handler RequestHandler) SubmitResult_(ctx context.Context, result *infra.ResultJob) (bool, error) {
	if result.Status == goodJob {

		sql := fmt.Sprintf("UPDATE jobs set status=2, seconds=%d, steps=%d,cycles=%d,score=%f where job_id=%d",
			result.Seconds, result.Steps, result.Cycles, result.Score, result.JobID)
		_, err := gDatabase.db.Exec(sql)
		if err != nil {
			log.Error(err)
			return false, err
		}

		go resultCallback(result)

	} else { // this worker wasn't up to the task. return the job to the pool
		sql := fmt.Sprintf("UPDATE jobs set status=0, worker_name='', instance_name='' where job_id=%d", result.JobID)
		_, err := gDatabase.db.Exec(sql)
		if err != nil {
			log.Error(err)
			return false, err
		}
	}

	return true, nil
}

func (handler RequestHandler) AddJob(ctx context.Context, agentID int32, worldID int32,
	agentCfg string, worldCfg string, priority int32, userID int32, note string) (int32, error) {
	sql := fmt.Sprintf("INSERT into jobs (user_id,priority,agent_id,world_id,agent_param,world_param,note) values (%d,%d,%d,%d,'%s','%s','%s')",
		userID, priority, agentID, worldID, agentCfg, worldCfg, note)
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
	// javascript doesn't support int64
	return int32(insertID), nil
}

// only allow you to delete unservered up jobs
func (handler RequestHandler) RemoveJob(ctx context.Context, jobID int32) (bool, error) {
	sql := fmt.Sprintf("DELETE from jobs where job_id=%d and status=0", jobID)
	_, err := gDatabase.db.Exec(sql)
	if err != nil {
		return false, err
	}
	return true, nil
}

func (handler RequestHandler) QueryJobs(ctx context.Context) ([]*infra.JobInfo, error) {
	return gDatabase.QueryJobs()
}

func (handler RequestHandler) GetBinInfo(ctx context.Context, binID int32) (*infra.BinInfo, error) {
	binInfo := gDatabase.GetBinInfo(binID)
	if binInfo == nil {
		return nil, errors.New("bin not found")
	}
	return binInfo, nil
}

func (handler RequestHandler) GetBinInfos(ctx context.Context, filterBy string) ([]*infra.BinInfo, error) {
	return gDatabase.GetBinInfos(filterBy)
}

func (handler RequestHandler) RunSQL(ctx context.Context, sql string) (string, error) {
	rows, err := gDatabase.db.Query(sql)
	if err != nil {
		log.Error(err)
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

// convert a single row to a map
// LATER: create a type for job row once the api is more stable
// but for now just map to strings, which is fine because we want to display them
func rowToMap(row *sql.Rows) (map[string]string, error) {
	columns, _ := row.Columns()
	values := make([]sql.RawBytes, len(columns))
	scanArgs := make([]interface{}, len(values))
	for i := range values {
		scanArgs[i] = &values[i]
	}

	res := make(map[string]string)

	err := row.Scan(scanArgs...)
	if err != nil {
		log.Error(err)
		return nil, err
	}

	var value string
	for i, col := range values {
		if col == nil {
			value = "NULL"
		} else {
			value = string(col)
		}
		res[columns[i]] = value
	}

	return res, nil
}

package main

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
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

func (handler RequestHandler) FetchWork(ctx context.Context, workerName string, instanceID int32) (*infra.Job, error) {
	return gDatabase.FetchWork(workerName, instanceID)
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
		_, err := gDatabase.UpdateGoodJob(result)
		if err != nil {
			log.Error(err)
			return false, err
		}

		go resultCallback(result)

	} else {
		// this worker wasn't up to the task. return the job to the pool
		_, err := gDatabase.UpdateFailedJob(result)
		if err != nil {
			log.Error(err)
			return false, err
		}
	}

	return true, nil
}

func (handler RequestHandler) AddJob(ctx context.Context, agentId int32, worldId int32,
	agentParam string, worldParam string, priority int32, userId int32, note string) (int32, error) {
	lastInsertID, err := gDatabase.AddJob(agentId, worldId, agentParam, worldParam, priority, userId, note)
	// javascript doesn't support int64
	return int32(lastInsertID), err
}

// RemoveJob only allow you to delete unservered up jobs
func (handler RequestHandler) RemoveJob(ctx context.Context, jobID int32) (bool, error) {
	return gDatabase.RemoveJob(jobID)
}

func (handler RequestHandler) QueryJobs(ctx context.Context, filterBy string) ([]*infra.JobInfo, error) {
	return gDatabase.QueryJobs(filterBy)
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

func (handler RequestHandler) RunSQL(ctx context.Context, query string) ([]map[string]string, error) {
	rows, err := gDatabase.db.Query(query)
	if err != nil {
		log.Error(err)
		return nil, err
	}

	res := []map[string]string{}

	for rows.Next() {
		m, err := rowToMap(rows)
		if err == nil {
			res = append(res, m)
		}
	}
	return res, nil
}

// convert a single row to a map
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

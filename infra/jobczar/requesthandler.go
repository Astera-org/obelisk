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

const (
	goodJob      int32 = 0
	jobFailed          = 1
	malformedJob       = 2
)

func (handler RequestHandler) FetchWork(ctx context.Context, workerName string, instanceName string) (*infra.Job, error) {
	println("job czar")

	job := infra.Job{}

	row := gDatabase.db.QueryRow("SELECT job_id,agent_name,world_name FROM jobs where status=0 order by priority desc LIMIT 1")

	err := row.Scan(&job.JobID, &job.AgentName, &job.WorldName)
	if err == sql.ErrNoRows {
		fmt.Println(err)
		return &job, errors.New("empty")
	}
	if err != nil {
		fmt.Println(err)
		return &job, errors.New("db error")
	}

	sql := fmt.Sprintf("UPDATE jobs set status=1, woker_name=`%s`, instance_name=`%s` where job_id=%d", workerName, instanceName, job.JobID)
	gDatabase.db.Exec(sql)

	return &job, nil
}


func (handler RequestHandler) SubmitResult_(ctx context.Context, result *infra.ResultWork) (bool, error) {

	println("SUBMIT RESULT")

	if result.Status == goodJob {
		sql := fmt.Sprintf("UPDATE jobs set status=2, cycles=%d,time_start=%d,time_end=%d,score=%f where job_id=%d", result.Cycles, result.TimeStart, result.TimeStop, result.Score, result.JobID)
		_, err := gDatabase.db.Exec(sql)
		if err != nil {
			fmt.Println(err)
			return false, err
		}
	} else { // this worker wasn't up to the task. return the job to the pool
		sql := fmt.Sprintf("UPDATE jobs set status=0, woker_name=``, instance_name=`` where job_id=%d", result.JobID)
		gDatabase.db.Exec(sql)
	}

	return true, nil
}

func (handler RequestHandler) AddJob(ctx context.Context, agentName string, worldName string,
	agentCfg string, worldCfg string, priority int32, userID int32) (int32, error) {
	println("job czar")

	sql := fmt.Sprintf("INSERT into jobs (user_id,priority,agent_name,world_name,agent_param,world_param) values (%d,%d,`%s`,`%s`,`%s`,`%s`)", userID, priority, agentName, worldName, agentCfg, worldCfg)
	_, err := gDatabase.db.Exec(sql)

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
func (handler RequestHandler) RemoveJob(ctx context.Context, jobID int32) (bool, error) {

	sql := fmt.Sprintf("DELETE from jobs where job_id=%d and status=0", jobID)
	_, err := gDatabase.db.Exec(sql)
	if err != nil {
		return false, err
	}
	return true, nil
}

func (handler RequestHandler) RunSQL(ctx context.Context, sql string) (string, error) {
	println("job czar")
	//rows, err := gDatabase.db.Query(sql)
	//if err != nil {
	//	fmt.Println(err)
	//	return "error", errors.New("db error")
	//}

	return "I love puppies!", nil

	//return printDBResult(rows), nil
}

func printDBResult(rows *sql.Rows) string {
	println("job czar")
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

package main

import (
	"context"
	"database/sql"

	"github.com/Astera-org/obelisk/infra/gengo/infra"
)

type RequestHandler struct {
	infra.JobCzar
}

func (agent RequestHandler) FetchWork(ctx context.Context, workerName string, instanceName string) (*infra.Job, error) {

	job := infra.Job{}

	rows, err := gDatabase.db.Query("SELECT job_id,model_name,env_name FROM jobs where status=0 order by priority desc LIMIT 1")
	if err != nil {
		fmt.Println(err)
		return &job, error("db error")
	}

	err = rows.Scan(&job.jobID, &job.agentName, &job.worldName)
	if err != nil {
		fmt.Println(err)
		return &job, error("empty")
	}

	gDatabase.db.Exec("UPDATE jobs set woker_name=$1, instance_name=$2 where job_id=$3",workerName,instanceName,job.jobID)

	return &job, nil
}

1: i32 jobID,
2: i32 cycles,
3: i32 timeStart,
4: i32 timeStop,
5: double score,
6: string workerName,
7: string instanceName

func (agent RequestHandler) submitResult(ctx context.Context, result infra.ResultWork) (bool, error) {
	
	err = gDatabase.db.Exec("UPDATE jobs set cycles=$1,time_start=$2,timeStop=$3,score=$4  where job_id=$5",result.cycles,result.timeStart,result.timeStop,result.score,result.jobID)
	if err != nil 
	{
		
	}

	return true, nil
}

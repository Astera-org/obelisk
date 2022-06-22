namespace go infra

// ..\thrift -r --gen go -out gengo/ infra.thrift


// Controller giving a Job to a worker
struct Job {
    1: i32 jobID,
    2: string agentName,
    3: string worldName,
    4: string agentCfg,
    5: string worldCfg
}

// I woudl call this WorkResult but
// thrift appends an _ if a typename ends in "result" ?
// status: 0-ok, 1-couldn't run job, 2- job malformed
struct ResultWork {
    1: i32 jobID,
    2: i32 status,
    3: i32 cycles,
    4: i32 seconds,
    6: double score,
    7: string workerName,
    8: string instanceName
}

service JobCzar {

    // workers call this when they are ready for a new Job
    Job fetchWork(1:string workerName,2:string instanceName);

    // workers call this when they are done with an assigned task
    // if doesn't return true the worker should try to tell it again that the work is complete
    bool submitResult(1:ResultWork result);

    i32 addJob(1:string agentName, 2:string worldName, 3:string agentCfg, 
        4:string worldCfg, 5:i32 priority, 6:i32 userID, 7:string note);

    string runSQL(1:string query);

    bool removeJob(1:i32 jobID);

    list<map<string, string>> queryJobs();
}

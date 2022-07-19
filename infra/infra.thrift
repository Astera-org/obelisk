namespace go infra

// ..\thrift -r --gen go -out gengo/ infra.thrift


// Controller giving a Job to a worker
struct Job {
    1: i32 jobID,
    2: i32 agentID,
    3: i32 worldID,
    4: string agentCfg,
    5: string worldCfg
}

// I would call this JobResult but
// thrift appends an _ if a typename ends in "result" ?
// status: 0-ok, 1-couldn't run job, 2- job malformed
struct ResultJob {
    1: i32 jobID,
    2: i32 status,
    3: i32 seconds,
    4: i32 steps, // number of total world steps this job took
    5: i32 cycles, // normalized amount of compute this job took
    6: double score,
    7: string workerName,
    8: string instanceName
}

// basically a single row of the jobs table (see database.sql)
struct JobInfo {
    1: i32 jobID,
    2: i32 userID,
    3: i32 searchID,
    4: i32 status,
    5: i32 priority,
    6: optional string callback,
    7: string timeAdded,
    8: i32 agentID,
    9: i32 worldID,
    10: string agentParam,
    11: string worldParam,
    12: string note,
    13: double bailThreshold,
    14: optional string workerName,
    15: optional string instanceName,
    16: optional string timeHanded,
    17: optional i32 seconds,
    18: optional i32 steps,
    19: optional i32 cycles,
    20: optional bool bailed,
    21: double score
}

// basically a single row of the binaries table (see database.sql)
struct BinInfo {
    1: i32 binID,
    2: string name,
    3: string version,
    4: string packageHash,
    5: string timeAdded,
    6: i32 type,
    7: i32 status
}

service JobCzar {

    // workers call this when they are ready for a new Job
    Job fetchWork(1:string workerName,2:string instanceName);

    // workers call this when they are done with an assigned task
    // if doesn't return true the worker should try to tell it again that the work is complete
    bool submitResult(1:ResultJob result);

    i32 addJob(1:i32 agentID, 2:i32 worldID, 3:string agentCfg, 
        4:string worldCfg, 5:i32 priority, 6:i32 userID, 7:string note);

    void fetchRunResults(1:i32 jobID);

    void appendNote(1:i32 jobID, 2:string note);

    BinInfo getBinInfo(1:i32 binID);

    list<BinInfo> getBinInfos(1:string filterBy);

    string runSQL(1:string query);

    bool removeJob(1:i32 jobID);

    list<JobInfo> queryJobs();
}

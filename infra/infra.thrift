namespace go infra

// Note: make sure the struct fields match the sql column names in database.sql.
// (the sql library lower cases the field names)
// That makes it easy for the sqlx library to fill in the corresponding fields when querying.

// Controller giving a Job to a worker
struct Job {
    1: i32 job_id,
    2: i32 agent_id,
    3: i32 world_id,
    4: string agent_param,
    5: string world_param
}

// I would call this JobResult but
// thrift appends an _ if a typename ends in "result" ?
// status: 0-working, 1-completed, 2-couldn't run job, 3- job malformed
struct ResultJob {
    1: i32 job_id,
    2: i32 status,
    3: i32 seconds,
    4: i32 steps, // number of total world steps this job took
    5: i32 targetSteps,
    6: i32 cycles, // normalized amount of compute this job took
    7: double score,
    8: string worker_name,
    9: i32 instance_id 
}

// basically a single row of the jobs table (see database.sql)
struct JobInfo {
    1: i32 job_id,
    2: i32 user_id,
    3: i32 search_id,
    4: i32 status,
    5: i32 priority,
    6: optional string callback,
    7: string time_added,
    8: i32 agent_id,
    9: i32 world_id,
    10: string agent_param,
    11: string world_param,
    12: string note,
    13: double bail_threshold,
    14: optional string worker_name,
    15: optional i32 instance_id,
    16: optional string worker_ip,
    17: optional string time_handed,
    18: optional i32 seconds,
    19: optional i32 steps,
    20: optional i32 cycles,
    21: optional bool bailed,
    22: double score
}

// basically a single row of the binaries table (see database.sql)
struct BinInfo {
    1: i32 bin_id,
    2: string name,
    3: string version,
    4: string package_hash,
    5: string time_added,
    6: i32 type,
    7: i32 status
}

service JobCzar {

    // workers call this when they are ready for a new Job
    Job fetchWork(1:string worker_name, 2:i32 instance_id);

    // workers call this when they are done with an assigned task
    // if doesn't return true the worker should try to tell it again that the work is complete
    bool submitResult(1:ResultJob result);

    i32 addJob(1:i32 agent_id, 2:i32 world_id, 3:string agent_param,
        4:string world_param, 5:i32 priority, 6:i32 user_id, 7:string note);

    void fetchRunResults(1:i32 job_id);

    bool updateNote(1:i32 job_id, 2:string note);

    BinInfo getBinInfo(1:i32 bin_id);

    list<BinInfo> getBinInfos(1:string filter_by);

    list<map<string, string>> runSQL(1:string query);

    bool removeJob(1:i32 job_id);

    list<JobInfo> queryJobs(1:string filter_by);

    // tell worker to stop job
    void stopJob(1:i32 job_id);
}

service WorkerService {
    void stopJob(1:i32 job_id);

    // send the results of a previous job to the binserver
    void sendResults(1:i32 job_id); 
}

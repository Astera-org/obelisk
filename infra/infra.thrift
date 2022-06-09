namespace go infra

// ..\thrift -r --gen cpp infra.thrift


// Controller giving a Job to a worker
struct Job {
    1: i32 jobID,
    2: string agentName,
    3: string worldName
}

// I woudl call this WorkResult but
// thrift appends an _ if a typename ends in "result" ?
struct ResultWork {
    1: i32 jobID,
    2: i32 cycles,
    3: i32 timeStart,
    4: i32 timeStop,
    5: double score,
    6: string workerName,
    7: string instanceName
}

service JobCzar {

    // workers call this when they are ready for a new Job
    Job fetchWork(1:string workerName,2:string instanceName);

    
    // workers call this when they are done with an assigned task
    // if doesn't return true the worker should try to tell it again that the work is complete
    bool submitResult(1:ResultWork result);

    i32 addJob(1:string agentName, 2:string worldName, 3:string agentCfg, 
        4:string worldCfg, 5:i32 priority, 6:i32 userID);

    string runSQL(1:string query);

    bool removeJob(1:i32 jobID);
}

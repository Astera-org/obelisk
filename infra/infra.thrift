namespace go infra

// ..\thrift -r --gen cpp infra.thrift


// Controller giving a Job to a worker
struct Job {
    1: i32 jobID,
    2: string agentName,
    3: string worldName
}


struct WorkResult {
    1: i32 jobID,
    2: i32 cycles,
    3: i32 timeStart,
    4: i32 timeStop,
    5: double score,
    11: string workerName,
    12: string instanceName
}

service JobCzar {

    // workers call this when they are ready for a new Job
    Job fetchWork(1:string workerName,2:string instanceName);

    
    // workers call this when they are done with an assigned task
    // if doesn't return true the worker should try to tell it again that the work is complete
    bool submitResult(1:WorkResult result);
}

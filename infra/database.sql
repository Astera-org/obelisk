
create database obelisk;
use obelisk;

#status (0-pending,1-working,2-complete,3-errored)

drop table jobs;
create table jobs (job_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY, user_id INT DEFAULT 0, 
    search_id INT DEFAULT 0, status INT DEFAULT 0, priority INT DEFAULT 0,
    time_added TIMESTAMP DEFAULT NOW(), 
    agent_id INT DEFAULT 0,
    world_id INT DEFAULT 0,
    agent_param VARCHAR(1024) default "", world_param VARCHAR(1024) default "",
    note VARCHAR(1024) default "",
    bail_threshold DOUBLE DEFAULT 0, 
    worker_name VARCHAR(10), instance_name VARCHAR(10),
    time_handed TIMESTAMP, seconds INT,
    cycles INT, bailed BOOLEAN, score DOUBLE DEFAULT 0);

create table cfgs (cfg_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY, cfg VARCHAR(1024));

create table users (user_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY, name VARCHAR(10));

#status (0-current,1-archived)
create table binaries (bin_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY, name VARCHAR(10), version VARCHAR(10), 
    hash VARCHAR(40), time_added TIMESTAMP DEFAULT NOW(), status INT DEFAULT 0);

####################

INSERT INTO jobs (agent_name,world_name,agent_param,world_param) values ("brain","fworld","GUI=false","GUI=false");


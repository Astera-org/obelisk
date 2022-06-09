
create database obelisk;
use obelisk;

#status (0-pending,1-working,2-complete,3-errored)

drop table jobs;
create table jobs (job_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY, user_id INT DEFAULT 0, 
    search_id INT DEFAULT 0, status INT DEFAULT 0, priority INT DEFAULT 0,
    time_added TIMESTAMP DEFAULT NOW(), 
    agent_name VARCHAR(10) NOT NULL,agent_version INT DEFAULT 0,
    world_name VARCHAR(10) NOT NULL,world_version INT DEFAULT 0,
    agent_param LONGBLOB, world_param LONGBLOB, 
    bail_threshold DOUBLE DEFAULT 0, 
    worker_name VARCHAR(10), instance_name VARCHAR(10),
    time_start TIMESTAMP DEFAULT 0, time_end TIMESTAMP DEFAULT 0,
    cycles INT, bailed BOOLEAN, score DOUBLE DEFAULT 0);

create table users (user_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY, user_name VARCHAR(10));

####################

INSERT INTO jobs (agent_name,world_name) values ("brain","fworld");


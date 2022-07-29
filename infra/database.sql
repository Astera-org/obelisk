drop database if exists obelisk;

create database obelisk;
use obelisk;

#status (0-pending,1-working,2-complete,3-errored)
#priority lower priority numbers take precedence
drop table if exists jobs;
create table jobs (job_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY, user_id INT DEFAULT 0,
    search_id INT DEFAULT 0, status INT DEFAULT 0, priority INT DEFAULT 0,
    callback VARCHAR(30), time_added TIMESTAMP DEFAULT NOW(),
    agent_id INT DEFAULT 0,
    world_id INT DEFAULT 0,
    agent_param VARCHAR(1024) DEFAULT "", world_param VARCHAR(1024) DEFAULT "",
    note VARCHAR(1024) DEFAULT "",
    bail_threshold DOUBLE DEFAULT 0,
    worker_name VARCHAR(10), instance_name VARCHAR(10),
    time_handed TIMESTAMP, seconds INT, steps INT,
    cycles INT, bailed BOOLEAN, score DOUBLE DEFAULT 0);


#type (0-agent,1-world)
#status (0-current,1-archived,2-validating,3-regressed)
drop table if exists binaries;
create table binaries (bin_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY, name VARCHAR(10), version VARCHAR(10), 
    package_hash VARCHAR(40), time_added TIMESTAMP DEFAULT NOW(), type INT DEFAULt 0, status INT DEFAULT 0);

create table notes (note_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,job_id INT,user_id INT,note_text VARCHAR(1024) );

drop table if exists cfgs;
drop table if exists users;
create table users (user_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY, name VARCHAR(10));



####################
# LATER
# create table cfgs (cfg_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY, cfg VARCHAR(1024));
# create table search (search_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,user_id INT, status INT,
#   priority INT, time_start TIMESTAMP DEFAULT NOW(), agent_id INT DEFAULT 0, world_id INT DEFAULT 0 );



#####################
INSERT INTO binaries (name, version, package_hash, type, status) VALUES ("protobrain", "1.0", "12345", 0, 0);
INSERT INTO binaries (name, version, package_hash, type, status) VALUES ("fworld", "1.0", "12345", 1, 0);
INSERT INTO jobs (agent_id,world_id,agent_param,world_param) values (1,2,"GUI=false","GUI=false");

INSERT INTO notes (job_id, user_id, note_text) VALUES (1, 0, "first note");
INSERT INTO notes (job_id, user_id, note_text) VALUES (1, 0, "second note");

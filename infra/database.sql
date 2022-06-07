
create database obelisk;
use obelisk;


drop table jobs;
create table jobs (job_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY, user_id INT DEFAULT 0, 
    search_id INT DEFAULT 0, status INT DEFAULT 0, priority INT DEFAULT 0,
    time_added TIMESTAMP DEFAULT NOW(), worker_name VARCHAR(10), instance_name VARCHAR(10),
    time_start TIMESTAMP DEFAULT 0, time_end TIMESTAMP DEFAULT 0,
    model_name VARCHAR(10),model_version INT DEFAULT 0,
    env_name VARCHAR(10),env_version INT DEFAULT 0,
    num_runs INT, num_trials INT, num_steps INT,
    model_param LONGBLOB, env_param LONGBLOB, bail BOOLEAN, score DOUBLE DEFAULT 0);

create table users (user_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY, user_name VARCHAR(10));


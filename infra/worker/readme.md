Sits on the cluster machines. 

1) fetches a job from the jobCzar
2) Creates the .cfg used for the spawns
3) Spawns the necessary processes
4) Read the spawn result report
5) Reports the result of the process to the jobCzar
6) delete the spawn result report


How does it know when a spawn is done:
- spawn msgs worker
- spawn writes to file. worker polls
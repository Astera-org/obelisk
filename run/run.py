# a simple helper script that starts both the model and the world

# TODO: start agent, wait for some output that says it's ready
# once ready start world, connect to server, run it

import subprocess

# start model first because that acts as a server so it has to be ready
result = subprocess.run(
    ["go run *.go"], text=True, shell=True, capture_output=True, cwd='../worlds/integrated/fworld'
)

print("stdout:", result.stdout)

# now that the model is ready start the world

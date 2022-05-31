import sys

import gym

from network.genpy.env.ttypes import Shape, SpaceSpec, ETensor, Action
from network.thrift_agent_client import setup_client


if __name__ == '__main__':

    env = gym.make('CartPole-v1')
    print(env.action_space)
    print(env.observation_space)

    agent = setup_client()


    action_spec = SpaceSpec(Shape(shape=[1]))
    observe_spec= SpaceSpec(Shape(shape=[2]))

    actionSpace = {"move": action_spec}
    observationSpace = {"world": observe_spec}

    for episode in range(1000):
        observation = env.reset()

        name = agent.init(actionSpace, observationSpace)
        print(f'new agent: {name}')

        past_reward = 0

        for t in range(1000):
            env.render()
            observations = {"world": ETensor(observe_spec.shape, values=observation), "past_reward": ETensor(Shape(shape=[1]), values=list([past_reward]))}

            actions = agent.step(observations, f'episode:{t}')
            #print(actions)
            action: Action = int(actions["move"].discreteOption)
            observation, past_reward, done, info = env.step(action)
            if done:
                agent.step({"done": ETensor(observe_spec.shape)}, "done") # Tell the agent to finish up
                print(f"Episode finished after {t+1} steps")
                break

    # TODO: add a close method to the environment
    # env.close()

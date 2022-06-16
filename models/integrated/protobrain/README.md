# Protobrain v1.0

Protobrain is an embodied rodent-level system's neuroscience model, focused on basic navigation in a virtual 2D world ("flat world" = FWorld), with a basic foraging task for food and water, which are depleted by time and activity, and replenished by eating / drinking.

Protobrain's cortical networks learn by predictive learning, initially by predicting the actions taken by a simulated reflexive-level subcortical system.  Then we'll introduce PFC / BG and hippocampal systems to support more strategic behavior and episodic memory.


# Learning logic

* predictive logic: SMA = current action, influences Super layers on time T, T+1 prediction = sensory outcome associated with that action (state updates happens at start of new cycle, after action is taken)

* key insight: if error-driven learning is only operative form of learning, that's all the model cares about, and it just doesn't stop to eat or drink!

# Known Issues

* The depth view scanner can see through non-H/V lines sometimes, if there is a "thin" diagonal aligned just so along its track.  use double-thick diagonal lines to be safe.

# TODO

* add gradual transition schedule to self-control during training..

* S1SP broken?

* LIP projects attention to ITP -- tune it

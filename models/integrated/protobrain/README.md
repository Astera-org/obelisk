# Emery v1.0

Emery is an embodied rodent-level system's neuroscience model, focused on basic navigation in a virtual 2D world ("flat world" = FWorld), with a basic foraging task for food and water, which are depleted by time and activity, and replenished by eating / drinking.

Emery's cortical networks learn by predictive learning, initially by predicting the actions taken by a simulated reflexive-level subcortical system.  Then we'll introduce PFC / BG and hippocampal systems to support more strategic behavior and episodic memory.

# Flat World

FWorld is a flat 2D world environment that can provide a minimal spatial navigational challenge for a simulation, without requiring any GPU rendering.  It is designed for developing system's level models of the many brain areas involved in spatial navigation.

The framework is basically grid world / minecraft style in 2D: there is a large grid represented as a 2D `int` Tensor where each cell can be painted with different block types, where 0 is empty space, and non-empty cells represent user-defined features such as "wall", "object1", "food", "water", etc.

When front adjacent cell is "food" and agent executes the "eat" action, food reward US is activated, and likewise for drink and water.  Cells could also have a "cover" such that "dig" needs to be executed, after which point food or water would be revealed, etc.

The agent is represented with a cell position and angle orientation, rodent-style without separate head or eye degrees of freedom, but with cat / primate-style forward-looking view (different options can be added later, including full rodent 360 deg with two side-facing eyes, etc).  

The first-person sensory state for the agent consists of:

* A dorsal-pathway full-field visual log-depth map of some angle of 1D FOV radial rendering from current location: rays are projected out from that location across an arc of e.g., 180 degrees, in e.g., 15 degree increments, and the depth to nearest non-empty block is represented as a pop-code value (log scaled).  This results in e.g., a 1 x 13 x 12 x 1 tensor with 13 angles around the arc and 12 values in the depth pop code.

* A ventral pathway "Fovea" bit pattern representing the cell contents closest to the directly-in-front cell.  This pattern could be low res at e.g., 36x36 and resolves wall vs. a few different types of landmarks.

* Proximal whisker / somatosensory sensor "ProxSoma" indicating contact with a surface along each of the 4 surrounding cells -- two bits per each cell, one for no-contact and the other for contact.

* Vestibular signal reflecting the delta-angle of rotation, as a pop-code (L, none, R).

* Interoceptive body state signals ("Inters") as pop codes that update in response to expenditure of effort, passage of time, and consumption of food / water.

* Optionally could include an olfactory gradient, but focusing on the visual modality initially.

There are 4 discrete movement actions, plus any additional optional interaction actions: eat, drink, dig, etc., all represented as bit patterns:

* Rotate L / R by fixed number of degrees (e.g., 15).  Rotation generates vestibular signal.

* Move Forward or Backward along current heading.  This keeps track of angle vector and increments discrete grid points according to closest such grid point.

* Actions can alter the environment state, creating additional opportunities for predictive learning to update

This environment thus supports a rich, extensible, ecologically-based framework in which to explore the temporally-extended pursuit of basic survival goals.

# Learning logic

* predictive logic: SMA = current action, influences Super layers on time T, T+1 prediction = sensory outcome associated with that action (state updates happens at start of new cycle, after action is taken)

* key insight: if error-driven learning is only operative form of learning, that's all the model cares about, and it just doesn't stop to eat or drink!

# Known Issues

* The depth view scanner can see through non-H/V lines sometimes, if there is a "thin" diagonal aligned just so along its track.  use double-thick diagonal lines to be safe.

# TODO

* add gradual transition schedule to self-control during training..

* S1SP broken?

* LIP projects attention to ITP -- tune it

"""
Generate pkl file for submission
"""

import os
import pickle
import torch

POLICY_PY_NAME = "policy_reduce_v2.py"
POLICY_CLASS_NAME = "ReducedModelV2"
# .pth file
CHECKPOINT_TO_SUBMIT = ""
OUT_NAME = CHECKPOINT_TO_SUBMIT + ".pkl"

# replace policy.py with your file
custom_policy_file = "reinforcement_learning/" + POLICY_PY_NAME
assert os.path.exists(custom_policy_file), "CANNOT find the policy file"
print(custom_policy_file)

# replace checkpoint with
checkpoint_to_submit = CHECKPOINT_TO_SUBMIT
assert os.path.exists(checkpoint_to_submit), "CANNOT find the checkpoint file"
assert checkpoint_to_submit.endswith(
    "_state.pth"
), "the checkpoint file must end with _state.pth"
print(checkpoint_to_submit)


def create_custom_policy_pt(policy_file, pth_file, out_name="my_submission.pkl"):
    assert out_name.endswith(".pkl"), "The file name must end with .pkl"
    with open(policy_file, "r") as f:
        src_code = f.read()

    # add the make_policy() function
    # YOU SHOULD CHECK the name of your policy (if not Baseline),
    # and the args that go into the policy
    src_code += f"""

class Config(nmmo.config.Default):
    PROVIDE_ACTION_TARGETS = True
    PROVIDE_NOOP_ACTION_TARGET = True
    MAP_FORCE_GENERATION = False
    TASK_EMBED_DIM = 4096
    COMMUNICATION_SYSTEM_ENABLED = False

def make_policy():
    from pufferlib.frameworks import cleanrl
    env = pufferlib.emulation.PettingZooPufferEnv(nmmo.Env(Config()))
    # Parameters to your model should match your configuration
    learner_policy = {POLICY_CLASS_NAME}(
        env,
        input_size=256,
        hidden_size=256,
        task_size=4096
    )
    return cleanrl.Policy(learner_policy)
  """
    state_dict = torch.load(pth_file, map_location="cpu")
    checkpoint = {
        "policy_src": src_code,
        "state_dict": state_dict,
    }
    with open(out_name, "wb") as out_file:
        pickle.dump(checkpoint, out_file)


create_custom_policy_pt(
    custom_policy_file,
    checkpoint_to_submit,
    out_name=OUT_NAME,
)

import gym_super_mario_bros
import imageio
from torch.utils.tensorboard import SummaryWriter
from gym.wrappers import ResizeObservation, GrayScaleObservation, FrameStack
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np

from torch.distributions import Categorical

from gym import Wrapper
import torch
from torch import nn

import os
JUMP_ACTIONS = {
    i for i, action in enumerate(SIMPLE_MOVEMENT) if 'A' in action
}

from datetime import datetime

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join("runs", run_id)
os.makedirs(run_dir, exist_ok=True)
tb_dir = os.path.join(run_dir, "tensorboard")
writer = SummaryWriter(log_dir=tb_dir)

dist_csv = os.path.join(run_dir, "distances.csv")
steps_csv = os.path.join(run_dir, "steps.csv")

eval_csv = os.path.join(run_dir, "eval.csv")
with open(eval_csv, "w", encoding="utf-8") as f:
    f.write("global_step,update,eval_mean_x,eval_max_x\n")


# distances.csv
with open(dist_csv, "w", encoding="utf-8") as f:
    f.write("life_idx,max_x_pos\n")

# steps.csv (log pe fiecare pas)
with open(steps_csv, "w", encoding="utf-8") as f:
    f.write(
        "global_step,update,step_in_update,life_idx,steps_in_life,"
        "action,entropy,log_prob,value,"
        "x_pos,dx,y_pos,dy,ground_y,stuck_steps,best_x_life,best_x_ever,"
        "lives,life_lost,terminated,truncated,forced_reset,done,"
        "score,dscore,last_jump_step,jump_age,is_jump,"
        "reward_total,terminal_reward,step_reward,"
        "r_prog,r_back,r_stagn,r_stuck,r_best_ever,r_death,r_timeout,"
        "r_jump_fail,r_jump_success,r_score,r_stomp,r_fall,stomp_event,death_cause,min_y_this_life\n"
    )


global_step = 0

####

###Agent

class Agent :
    def __init__(self,model,device=None):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)


    @torch.no_grad()
    def selectAction(self,state):
        stateNP = np.array(state, copy=False).astype(np.float32) / 255.0
        stateT = torch.tensor(stateNP, dtype=torch.float32, device=self.device).unsqueeze(0)

        logits,value=self.model(stateT)
        dist = Categorical(logits=logits)
        actionT=dist.sample()
        logProbT=dist.log_prob(actionT)
        entropyT=dist.entropy()

        return (
            int(actionT.item()),
            float(logProbT.item()),
            float(value.squeeze(-1).item()),
            float(entropyT.item()),
        )




### End Agent

### RolloutBuffer

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def add(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()



def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95, last_value=0.0):

    advantages = []
    gae = 0.0

    values = values + [float(last_value)]

    for t in reversed(range(len(rewards))):
        delta = (
            rewards[t]
            + gamma * values[t + 1] * (1 - dones[t])
            - values[t]
        )

        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    returns = [
        adv + val
        for adv, val in zip(advantages, values[:-1])
    ]

    return (
        torch.tensor(advantages, dtype=torch.float32),
        torch.tensor(returns, dtype=torch.float32),
    )



### End Rollout buffer

### PPoUpdate
def ppo_update(
    model,
    optimizer,
    states,
    actions,
    old_log_probs,
    returns,
    advantages,
    clip_eps=0.1,
    value_coef=0.5,
    entropy_coef=0.02,
    epochs=4,
    batch_size=128,
    max_grad_norm=0.5,
):
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    device = model.policyHead.weight.device

    states_np = np.array(states, dtype=np.float32) / 255.0  # imagini 0..1
    actions_t = torch.tensor(actions, dtype=torch.long, device=device)
    old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32, device=device)

    dataset_size = len(states_np)
    indices = np.arange(dataset_size)

    if not torch.is_tensor(returns):
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
    else:
        returns_t = returns.to(device)

    if not torch.is_tensor(advantages):
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
    else:
        advantages_t = advantages.to(device)

    for _ in range(epochs):
        np.random.shuffle(indices)

        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            batch_states = torch.tensor(states_np[batch_idx], dtype=torch.float32, device=device)
            batch_actions = actions_t[batch_idx]
            batch_old_log_probs = old_log_probs_t[batch_idx]
            batch_returns = returns_t[batch_idx]
            batch_advantages = advantages_t[batch_idx]

            logits, values = model(batch_states)
            values = values.squeeze(-1)

            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = (batch_returns - values).pow(2).mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()



###End PPo update

### cNN
class NeuronalNetwork (nn.Module):
    def __init__(self,input_dim,n_actions):
        super().__init__()

        self.cnn=nn.Sequential(
            nn.Conv2d(input_dim[0],32,kernel_size=8,stride=4),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=4,stride=2),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,stride=1),
            nn.ReLU(),
        )
        conv_out_size = self.getConvOut(input_dim)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
        )
        self.policyHead=nn.Linear(512,n_actions)
        self.valueHead=nn.Linear(512,1)

    def getConvOut(self,shape):
        o=self.cnn(torch.zeros(1,*shape))
        return int(np.prod(o.size()))

    def forward(self, x):

        features = self.cnn(x)
        features = self.fc(features)

        logits = self.policyHead(features)
        value = self.valueHead(features)

        return logits, value


###End ofcNN



### Wrapper

class SkipFrame(Wrapper):
    def __init__(self,env,skip=4):
        super().__init__(env)
        self.skip=skip

    def step(self,action):
        totalReward=0
        info={}
        terminated=False
        truncated=False
        for i in range(self.skip):
            nextState, reward, terminated, truncated, info = self.env.step(action)
            totalReward+=reward
            if terminated or truncated:
                break
        return nextState,totalReward,terminated,truncated,info

def applyWrappers(env):
    env=SkipFrame(env,skip=4)
    env=ResizeObservation(env,shape=(84,84))
    env=GrayScaleObservation(env)
    env=FrameStack(env,num_stack=4,lz4_compress=False)
    return env

### End of Wrapper



#### Main LoopF

env = gym_super_mario_bros.make(
    "SuperMarioBros-v0",
    apply_api_compatibility=True,
    render_mode=None
)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = applyWrappers(env)


eval_env = gym_super_mario_bros.make(
    "SuperMarioBros-v0",
    apply_api_compatibility=True,
    render_mode = None
)
eval_env = JoypadSpace(eval_env, SIMPLE_MOVEMENT)
eval_env = applyWrappers(eval_env)


model = NeuronalNetwork(
    env.observation_space.shape,
    env.action_space.n
)
agent = Agent(model)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)


ROLLOUT_STEPS =4096
TOTAL_UPDATES = 100000

RESET_ON_LIFE_LOST = True
state, info = env.reset()
life_idx = 0
steps_in_life = 0
best_x_ever = 0
stuck_steps = 0
no_progress_steps = 0
prev_x = int(info.get("x_pos", 0))
best_x_this_life = prev_x
prev_score = info.get("score", 0)
prev_life = None
episode_reward = 0.0
prev_y = info.get("y_pos", 0)
ground_y = prev_y
best_eval_mean = -1e9
prev_time = info.get("time", 0)
best_eval_max = -1.0
prev_lives = info.get("life", info.get("lives", None))
last_jump_step = -10_000
in_gap = False
gap_start_x = 0
gap_min_y = 9999
min_y_this_life = int(info.get("y_pos", 0))

#### ev policy

def evaluate_policy(eval_env, agent, episodes=5, max_steps=6000):
    agent.model.eval()
    scores = []

    for _ in range(episodes):
        obs, info = eval_env.reset()
        terminated = False
        truncated = False
        best_x = 0

        for _ in range(max_steps):
            obs_np = np.array(obs, copy=False).astype(np.float32) / 255.0
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=agent.device).unsqueeze(0)

            with torch.no_grad():
                logits, value = agent.model(obs_t)
                action = int(torch.argmax(logits, dim=-1).item())

            obs, _, terminated, truncated, info = eval_env.step(action)
            best_x = max(best_x, info.get("x_pos", 0))

            if terminated or truncated:
                break

        scores.append(best_x)

    agent.model.train()
    return float(np.mean(scores)), float(np.max(scores))
#### end ev policy

###record
def record_best_run(video_path, agent, episodes=1, max_steps=6000):
    rec_env = gym_super_mario_bros.make(
        "SuperMarioBros-v0",
        apply_api_compatibility=True,
        render_mode="rgb_array"
    )
    rec_env = JoypadSpace(rec_env, SIMPLE_MOVEMENT)
    rec_env = applyWrappers(rec_env)

    writer = imageio.get_writer(video_path, fps=60)
    agent.model.eval()
    try:
        for _ in range(episodes):
            obs, info = rec_env.reset()
            terminated = False
            truncated = False

            for _ in range(max_steps):
                frame = rec_env.render()
                writer.append_data(frame)

                obs_np = np.array(obs, copy=False).astype(np.float32) / 255.0
                obs_t = torch.tensor(obs_np, dtype=torch.float32, device=agent.device).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = agent.model(obs_t)
                    action = int(torch.argmax(logits, dim=-1).item())

                obs, _, terminated, truncated, info = rec_env.step(action)
                if terminated or truncated:
                    frame = rec_env.render()
                    writer.append_data(frame)
                    break
    finally:
        writer.close()
        rec_env.close()
        agent.model.train()


###end recod

for update in range(TOTAL_UPDATES):
    buffer = RolloutBuffer()

    for step in range(ROLLOUT_STEPS):
        action, log_prob, value, entropy = agent.selectAction(state)
        next_state, env_reward, terminated, truncated, info = env.step(action)

        flag_get = bool(info.get("flag_get", False))
        done = terminated or truncated or flag_get

        forced_reset = False
        steps_in_life += 1


        global_step += 1

        if action in JUMP_ACTIONS:
            last_jump_step = global_step

        is_jump = int(action in JUMP_ACTIONS)
        jump_age = global_step - last_jump_step

        x_pos = int(info.get("x_pos", prev_x))
        dx = x_pos - prev_x

        y_pos = int(info.get("y_pos", prev_y))
        min_y_this_life = min(min_y_this_life, y_pos)

        if x_pos > best_x_this_life + 1:  # progres real
            no_progress_steps = 0
        else:
            no_progress_steps += 1

        lives = info.get("life", info.get("lives", None))
        life_lost = (prev_lives is not None and lives is not None and lives < prev_lives)
        prev_lives = lives



        if life_lost:
            done = True
            terminated = False
            truncated = False
            forced_reset = False
        if done:
            x_pos = prev_x
            dx = 0

        r_idle = 0.0
        if no_progress_steps > 10:
            r_idle -= 0.002



        score = info.get("score", prev_score)
        dscore = score - prev_score
        prev_score = score
        stomp_event = int(dscore >= 100)

        r_score = 0.0
        if dscore > 0:
            r_score = 0.001 * dscore

        r_stomp = 0.0
        if dscore >= 100 and (global_step - last_jump_step) <= 30:
            r_stomp += 2.0

        r_prog = r_back = r_stagn = r_stuck  = r_best_ever = r_death = r_timeout = r_jump_fail = 0.0
        if dx < -2 and stuck_steps < 10:
            r_back -= 0.02 * (-dx)

        r_prog = 0.0
        if dx > 0:
            r_prog += 0.01 * dx

        if action in JUMP_ACTIONS and dx <= 0 and no_progress_steps < 8:
            r_jump_fail = -0.01
        else:
            r_jump_fail = 0.0

        if action in JUMP_ACTIONS and dx > 0:
            r_jump_success = 0.01
        else:
            r_jump_success = 0.0

        if no_progress_steps > 600:
            forced_reset = True
            done = True
            truncated = False
            terminated = False
            r_stuck -= 2.0

        stuck_steps = no_progress_steps

        r_timeout = 0.0
        if truncated and (not forced_reset):
            r_timeout -= 15.0

        if stuck_steps > 30:
            r_stuck -= 0.02
        if stuck_steps > 60:
            r_stuck -= 0.05
        if stuck_steps > 120:
            r_stuck -= 0.10
        if stuck_steps >300:
            r_stuck -=0.30

        if x_pos > best_x_this_life:
            best_x_this_life = x_pos

        if x_pos > best_x_ever + 10:
            best_x_ever = x_pos
            r_best_ever += 0.1

        dy = y_pos - prev_y
        prev_y = y_pos

        if abs(dy) <= 1 and y_pos > 0:
            ground_y = 0.95 * ground_y + 0.05 * y_pos

        r_fall = 0.0

        if dy < -6 and y_pos < (ground_y - 30):
            r_fall -= 0.05


        t = info.get("time", prev_time)
        dt = prev_time - t
        prev_time = t

        if done and (not forced_reset) and y_pos <= 20:
            r_fall -= 1.0

        r_gap = 0.0

        GAP_DEPTH = 45
        GAP_MIN_ADV = 35
        GROUND_EPS = 10

        if (not in_gap) and (y_pos < (ground_y - GAP_DEPTH)):
            in_gap = True
            gap_start_x = x_pos
            gap_min_y = y_pos

        if in_gap:
            if y_pos < gap_min_y:
                gap_min_y = y_pos

        is_on_ground = (abs(dy) <= 2) and (abs(y_pos - ground_y) < GROUND_EPS)

        if in_gap and is_on_ground:
            if (x_pos - gap_start_x) >= GAP_MIN_ADV:
                r_gap += 1.0
            in_gap = False

        if done and (not forced_reset):
            r_death -= 8.0

        step_reward = (
                r_prog +
                r_back +
                r_stuck +
                r_jump_fail +
                r_jump_success +
                r_score +
                r_stomp+
                r_best_ever +
                r_fall +
                r_gap +
                r_idle
        )

        step_reward = float(np.clip(step_reward, -5.0, 5.0))

        terminal_reward = 0.0

        death_cause = 0
        if done:
            y_for_death = min_y_this_life if life_lost else int(y_pos)
            if y_for_death <= 20:
                death_cause = 1
            else:
                death_cause = 2

        r_enemy_death = 0.0
        if done and (not forced_reset) and death_cause == 2:
            r_enemy_death -= 2.0

        if done and (not forced_reset):
            terminal_reward += r_death+ r_enemy_death

        if truncated and (not forced_reset):
            terminal_reward += r_timeout

        if flag_get:
            terminal_reward += 50.0

        reward = step_reward + terminal_reward

        if global_step % 100 == 0:
            writer.add_scalar("reward/total_reward", reward, global_step)
            writer.add_scalar("reward/step_reward", step_reward, global_step)
            writer.add_scalar("reward/terminal_reward", terminal_reward, global_step)

            writer.add_scalar("reward/r_prog", r_prog, global_step)
            writer.add_scalar("reward/r_back", r_back, global_step)
            writer.add_scalar("reward/r_stuck", r_stuck, global_step)
            writer.add_scalar("reward/r_idle", r_idle, global_step)
            writer.add_scalar("reward/r_jump_fail", r_jump_fail, global_step)
            writer.add_scalar("reward/r_jump_success", r_jump_success, global_step)
            writer.add_scalar("reward/r_score", r_score, global_step)
            writer.add_scalar("reward/r_stomp", r_stomp, global_step)
            writer.add_scalar("reward/r_fall", r_fall, global_step)
            writer.add_scalar("reward/r_gap", r_gap, global_step)

            writer.add_scalar("debug/x_pos", x_pos, global_step)
            writer.add_scalar("debug/dx", dx, global_step)
            writer.add_scalar("debug/y_pos", y_pos, global_step)
            writer.add_scalar("debug/stuck_steps", stuck_steps, global_step)
            writer.add_scalar("debug/no_progress_steps", no_progress_steps, global_step)
            writer.add_scalar("debug/entropy_sample", entropy, global_step)

        if (global_step % 10 == 0) or done or life_lost or forced_reset:
            with open(steps_csv, "a", encoding="utf-8") as f:
                f.write(
                    f"{global_step},{update},{step},{life_idx},{steps_in_life},"
                    f"{action},{entropy:.6f},{log_prob:.6f},{value:.6f},"
                    f"{x_pos},{dx},{y_pos},{dy},{ground_y:.2f},{stuck_steps},{best_x_this_life},{best_x_ever},"
                    f"{lives},{int(life_lost)},{int(terminated)},{int(truncated)},{int(forced_reset)},{int(done)},"
                    f"{score},{dscore},{last_jump_step},{jump_age},{is_jump},"
                    f"{reward:.6f},{terminal_reward:.6f},{step_reward:.6f},"
                    f"{r_prog:.6f},{r_back:.6f},{r_stagn:.6f},{r_stuck:.6f},{r_best_ever:.6f},{r_death:.6f},{r_timeout:.6f},"
                    f"{r_jump_fail:.6f},{r_jump_success:.6f},{r_score:.6f},{r_stomp:.6f},{r_fall:.6f},{int(stomp_event)},{int(death_cause)},{min_y_this_life}\n"
                )

        buffer.add(
            state=state,
            action=action,
            reward=reward,
            done=done,
            log_prob=log_prob,
            value=value
        )

        prev_x = x_pos

        state = next_state

        if done:
            life_idx += 1
            writer.add_scalar("life/max_x_this_life", best_x_this_life, life_idx)
            writer.add_scalar("life/steps_in_life", steps_in_life, life_idx)


            with open(dist_csv, "a", encoding="utf-8") as f:
                f.write(f"{life_idx},{best_x_this_life}\n")

            steps_in_life = 0
            no_progress_steps = 0
            stuck_steps = 0

            state, info = env.reset()
            min_y_this_life = int(info.get("y_pos", 0))

            prev_x = int(info.get("x_pos", 0))
            prev_y = int(info.get("y_pos", 0))
            prev_score = int(info.get("score", 0))
            prev_time = int(info.get("time", 0))
            prev_lives = info.get("life", info.get("lives", None))

            ground_y = prev_y
            best_x_this_life = prev_x

            last_jump_step = -10_000
            in_gap = False
            gap_start_x = 0
            gap_min_y = 9999

            continue

    with torch.no_grad():
        state_np = np.array(state, copy=False).astype(np.float32) / 255.0
        state_t = torch.tensor(state_np, dtype=torch.float32, device=agent.device).unsqueeze(0)
        _, last_v = agent.model(state_t)
        last_value = float(last_v.squeeze(-1).item())

    if len(buffer.dones) > 0 and buffer.dones[-1]:
        last_value = 0.0

    advantages, returns = compute_gae(
        rewards=buffer.rewards,
        values=buffer.values,
        dones=buffer.dones,
        last_value=last_value
    )

    if update < 50:
        ent = 0.02
    elif update < 100:
        ent = 0.01
    else:
        ent = 0.003

    ppo_update(
        model=model,
        optimizer=optimizer,
        states=buffer.states,
        actions=buffer.actions,
        old_log_probs=buffer.log_probs,
        returns=returns,
        advantages=advantages,
        entropy_coef=ent
    )

    if update % 10 == 0:
        avg_reward = sum(buffer.rewards) / len(buffer.rewards)
        print(f"Update {update} | Avg reward: {avg_reward:.2f}")
        writer.add_scalar("train/avg_reward", avg_reward, update)
        writer.add_scalar("train/entropy_coef", ent, update)
    if update % 20 == 0:
        mean_x, max_x = evaluate_policy(eval_env, agent, episodes=5)
        print(f"[EVAL] update={update} mean_x={mean_x:.1f} max_x={max_x:.1f}")
        writer.add_scalar("eval/mean_x", mean_x, update)
        writer.add_scalar("eval/max_x", max_x, update)
        with open(eval_csv, "a", encoding="utf-8") as f:
            f.write(f"{global_step},{update},{mean_x:.1f},{max_x:.1f}\n")
            f.flush()

        if max_x > best_eval_max + 1:
            best_eval_max = max_x
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))

            video_path = os.path.join(run_dir, f"best_run_update_{update}_max_{int(max_x)}.mp4")
            record_best_run(video_path, agent, episodes=1)
        writer.add_scalar("eval/best_eval_max", best_eval_max, update)

env.close()
eval_env.close()
writer.close()
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import numpy as np
from dataclasses import dataclass
from typing import Optional
from sortedcontainers import SortedList
from collections import defaultdict 
import time
from sortedcontainers import SortedList
import json
import os
import operator 
import random

@dataclass
class KillStats:
    kill_time:float
    kill_pos:float
    kill_index:int

@dataclass
class Bullet:
    index:int
    speed:float
    kill_stats:Optional[KillStats] = None

    @classmethod
    def from_data(cls, data):
        return cls(index=data[0], speed=data[1])

    def __lt__(self, other):
         return self.index < other.index
    
    def dict(self):
        d = dict(self.__dict__)
        if self.kill_stats is not None:
            d['kill_stats'] = dict(self.kill_stats.__dict__)
        return d

def get_kill_stats(bullet, target_bullet):
    if bullet.speed <= target_bullet.speed:
        return None
    
    dx = (bullet.index - target_bullet.index) * target_bullet.speed
    dv = bullet.speed - target_bullet.speed
    dt = dx / dv
    
    kill_time = bullet.index + dt
    if target_bullet.kill_stats is not None and target_bullet.kill_stats.kill_time < kill_time:
        return None
    
    kill_pos = dt * bullet.speed

    return KillStats(kill_time=kill_time, kill_pos=kill_pos, kill_index=target_bullet.index)

def plot_front_diagram(data, N, ax):
    speed_data = data['speed_data']
    front = Bullet(index=-1, speed=speed_data[ALIVE])
    plot_bullet(bullet=front, t_2=N, linestyle=':', color='blue')

    lb_interceptor, ub_interceptor = interceptors(N=N, survivors=data['survivors'], 
        still_alive=data['still_alive'])

    front.speed = speed_data[UNDOOMED_LB]
    lb_interceptor.kill_stats = get_kill_stats(bullet=lb_interceptor, target_bullet=front)
    plot_collision(bullet_r=lb_interceptor, bullet_l=front, cutoff_right=False, N=None, color='purple', linestyle=':')

    front.speed = speed_data[UNDOOMED_UB]
    ub_interceptor.kill_stats = get_kill_stats(bullet=ub_interceptor, target_bullet=front)
    plot_collision(bullet_r=ub_interceptor, bullet_l=front, cutoff_right=False, N=None, color='black', linestyle=':')

    front.speed = speed_data[NONTHREATENED]
    plot_bullet(bullet=front, t_2=ax.get_xlim()[1], linestyle=':', color='red')

def plot_collision(bullet_r, bullet_l, cutoff_right, N, color=None, linestyle='-'):
    t_2 = bullet_r.kill_stats.kill_time
    if cutoff_right and t_2 > N:
        t_2 = N
        x_2 = None
    else:
        x_2 = bullet_r.kill_stats.kill_pos
    
    line = plot_bullet(bullet=bullet_r, t_2=t_2, x_2=x_2, color=color, linestyle=linestyle)
    plot_bullet(bullet=bullet_l, t_2=t_2, x_2=x_2, color=line.get_color(), linestyle=linestyle)

def plot_bullet(bullet, t_2, color=None, x_2=None, linestyle='-'):
    if x_2 is None:
        x_2 = (t_2 - bullet.index) * bullet.speed
    return plt.plot([bullet.index, t_2], [0, x_2], linestyle=linestyle, color=color)[0]


def plot_bullet_process(bullets, N=None, nested=True, 
                        cutoff_right=True, data=None, to_time=False):
    start = time.time()

    # all_colors = [k for k,v in pltc.cnames.items()]
    if N is None:
        N = len(bullets)
    cmap = plt.cm.get_cmap("hsv", N+1)
    ax = plt.gca()
    slain = set()
    bullet_to_index = {b.index: i for i, b in enumerate(bullets)}
    list_index = len(bullets) - 1
    survivors = []

    while list_index >= 0:
        bullet = bullets[list_index]
        if bullet.index in slain:
            list_index -= 1
            continue
            
        if bullet.kill_stats is None:
            survivors.append(bullet)
            list_index -= 1

        else:
            slain_index = bullet.kill_stats.kill_index
            slain.add(slain_index)
            slain_list_index = bullet_to_index[slain_index]
            slain_bullet = bullets[slain_list_index]
            plot_collision(bullet_r=bullet, bullet_l=slain_bullet, cutoff_right=cutoff_right, N=N)
            if nested:
                list_index -= 1
            else:
                list_index = slain_list_index - 1
    
    if data is not None:
        plot_front_diagram(data=data, N=N, ax=ax)

    for survivor in survivors:
        t_2 = N if cutoff_right else ax.get_xlim()[1]
        plot_bullet(bullet=survivor, t_2=t_2, color='green', linestyle=':')
    if not cutoff_right:
        plt.plot([N, N], [0, ax.get_ylim()[1]], linestyle="--", color='yellow')

    if to_time:
        print('draw_bullet_process: (hh:mm:ss.ms) {}'.format(time.time() - start))

def draw_bullet_process(bullets, N=None, nested=True, cutoff_right=True, to_time=False):
    plot_bullet_process(bullets=bullets, N=N, nested=nested, cutoff_right=cutoff_right, to_time=to_time)
    plt.show()

def get_collision_diagram(bullets):
    collision_diagram = set()
    N = len(bullets)
    slain = set()

    for index in reversed(range(N)):
        bullet = bullets[index]
        if index in slain or bullet.kill_stats is None:
            continue
        else:
            slain_index = bullet.kill_stats.kill_index
            # if bullet.kill_stats.kill_time < N - 1:
            collision_diagram.add((slain_index, index))
            slain.add(slain_index)

    return collision_diagram

# bullet_r == bullet_l if it's a survivor
def update_maxes(bullet_r, bullet_l, maxes, lb_interceptor, ub_interceptor, N):
    kill_stats = bullet_r.kill_stats
    if kill_stats is None:
        nt_speed = bullet_r.speed
    else:
        t_2 = kill_stats.kill_time
        x_2 = kill_stats.kill_pos
        nt_speed = x_2 / (t_2 + 1)
    
    if nt_speed <= maxes[ALIVE]:
        return

    maxes[NONTHREATENED] = max(nt_speed, maxes[NONTHREATENED])
    
    if kill_stats is not None and t_2 <= N:
        alive_speed, undoomed_lb_speed, undoomed_ub_speed = nt_speed, nt_speed, nt_speed
    else:
        alive_speed = bullet_l.speed * (N - bullet_l.index)/(N + 1)

        lb_kill_stats = get_kill_stats(bullet=lb_interceptor, target_bullet=bullet_l)
        lb_t2 = lb_kill_stats.kill_time
        if kill_stats is not None and lb_t2 >= t_2:
            undoomed_lb_speed, undoomed_ub_speed = nt_speed, nt_speed
        else:
            undoomed_lb_speed = lb_kill_stats.kill_pos / (lb_t2 + 1)
            ub_kill_stats = get_kill_stats(bullet=ub_interceptor, target_bullet=bullet_l)
            undoomed_ub_speed = ub_kill_stats.kill_pos / (ub_kill_stats.kill_time + 1)

    maxes[ALIVE] = max(alive_speed, maxes[ALIVE])
    maxes[UNDOOMED_LB] = max(undoomed_lb_speed, maxes[UNDOOMED_LB])
    maxes[UNDOOMED_UB] = max(undoomed_ub_speed, maxes[UNDOOMED_UB])

def interceptors(N, survivors, still_alive):
    lb_interceptor = Bullet(index=N, speed=1)
    ub_interceptor = Bullet(index=(N + len(still_alive) - 1), speed=1)
    return lb_interceptor, ub_interceptor

def init_maxes():
    return {speed_str: 0 for speed_str in FRONT_SPEEDS}

def get_front_survival_speeds(bullets, N, to_time=False):
    start = time.time()

    survivors, still_alive = get_survivors(bullets=bullets)
    lb_interceptor, ub_interceptor = interceptors(N=N, survivors=survivors, still_alive=still_alive)
    maxes = init_maxes()
    N = len(bullets)
    if len(survivors) > 0:
        update_maxes(bullet_r=survivors[0], bullet_l=survivors[0], maxes=maxes, 
                    lb_interceptor=lb_interceptor, ub_interceptor=ub_interceptor, N=N)
        index = survivors[0].index
    else:
        index = N-1

    while index >= 0:
        bullet = bullets[index]
        if bullet.kill_stats is None:
            index -= 1
        else:
            slain_bullet = bullets[bullet.kill_stats.kill_index]
            update_maxes(bullet_r=bullet, bullet_l=slain_bullet, maxes=maxes, 
                    lb_interceptor=lb_interceptor, ub_interceptor=ub_interceptor, N=N)
            
            index = slain_bullet.index - 1
    
    assert maxes[ALIVE] <= maxes[UNDOOMED_LB] <= maxes[UNDOOMED_UB] <= maxes[NONTHREATENED]

    if to_time:
        print('get_front_survival_speeds: (hh:mm:ss.ms) {}'.format(time.time() - start))

    return maxes

def plot_potential_survivors(potential_survivors, to_time=False):
    start = time.time()
    
    x_vals = []
    y_vals = []
    for bullet in potential_survivors:
        x_vals.append(bullet.index)
        y_vals.append(bullet.speed)
    plt.scatter(x_vals, y_vals)
    plt.show()

    if to_time:
        print('plot_potential_survivors: (hh:mm:ss.ms) {}'.format(time.time() - start))

MILLION = int(1e6)
def run_bullet_process(N, bullets = [], speeds = None, to_time=True):
    start = time.time()
    milestone = start
    if speeds is None:
        speeds = []
    speeds = np.concatenate((speeds, np.random.rand(N-len(speeds))))
    
    bullets = [Bullet(index=0, speed = speeds[0])]
    
    for index in range(1,N):
        if index % MILLION == 0:
            print("{:.1e}".format(index), time.time() - milestone, time.time() - start)
            milestone = time.time()
        bullet = Bullet(index=index, speed = speeds[index])
        bullets.append(bullet)
        target = index-1
        while True:
            if target < 0:
                break
                
            kill_stats = get_kill_stats(bullet=bullet, target_bullet=bullets[target])
            if kill_stats is not None:
                bullet.kill_stats = kill_stats
                break
    
            target_kill_stats = bullets[target].kill_stats
            if target_kill_stats is None:
                break
            else:
                target = target_kill_stats.kill_index - 1

    if to_time:
        print('run_bullet_process: (hh:mm:ss.ms) {}'.format(time.time() - start))
        
    return bullets

DATA_DIR = 'data'
def get_fname(N, seed, prefix='', just_suffix=False):
    return f"{DATA_DIR}/{prefix}N-{N}_seed-{seed}.json"

def plot_cdf(data, label):
    x, CDF_counts = np.unique(data, return_counts=True)
    y = np.cumsum(CDF_counts) / np.sum(CDF_counts)
    plt.plot(x,y, label=label)

def experiment(N, seed=None, speeds=None, prefix='', find_collisions=False, save=True, to_time=True):
    if seed is None:
        seed = np.random.randint(1e9)
    np.random.seed(seed)
    bullets = run_bullet_process(N=N, speeds=speeds, to_time=to_time)
    speed_data = get_front_survival_speeds(bullets=bullets, N=N, to_time=to_time)
    survivors, still_alive = get_survivors(bullets=bullets, to_time=to_time)
    collision_diagram = None
    if find_collisions:
        collision_diagram = get_collision_diagram(bullets=bullets)

    data = package_data(N=N, seed=seed, speed_data=speed_data, 
        survivors=survivors, still_alive=still_alive, collision_diagram=collision_diagram)
    if save:
        save_data(data=data, prefix=prefix)

    return data, bullets

def rerun_bp(survivors, still_alive, to_time=False):
    start = time.time()

    bullets = sorted(list(survivors) + list(still_alive), key=operator.attrgetter('index'))
    bullet_idx_to_list_idx = {}

    for i, bullet in enumerate(bullets):
        bullet_idx_to_list_idx[bullet.index] = i
        target_list_index = i-1
        
        while target_list_index >= 0:
            target_bullet=bullets[target_list_index]
            kill_stats = get_kill_stats(bullet=bullet, target_bullet=target_bullet)
            if kill_stats is not None:
                bullet.kill_stats = kill_stats
                break

            target_kill_stats = target_bullet.kill_stats
            if target_kill_stats is None:
                break
            else:
                target_list_index = bullet_idx_to_list_idx[target_kill_stats.kill_index] - 1

    if to_time:
        print('run_bullet_process: (hh:mm:ss.ms) {}'.format(time.time() - start))
        
    return bullets

def get_survivors(bullets, to_time=False):
    start = time.time()
    N = len(bullets)
    survivors = []
    still_alive = []
    still_alive_idxs = set()
    index = N - 1
    while index >= 0:
        bullet = bullets[index]

        if index in still_alive_idxs:
            still_alive.append(bullet)
            index -= 1
            continue
        
        kill_stats = bullet.kill_stats
        if kill_stats is None:
            survivors.append(bullet)
            still_alive.append(bullet)
            index -= 1
            continue

        if kill_stats.kill_time > N - 1:
            still_alive.append(bullet)
            still_alive_idxs.add(kill_stats.kill_index)
            index -= 1
        else:
            index = kill_stats.kill_index - 1
        
    if to_time:
        print('get_survivors: (hh:mm:ss.ms) {}'.format(time.time() - start))
    return list(reversed(survivors)), list(reversed(still_alive))

def dictify_bullets(bullets):
    return [bullet.dict() for bullet in bullets]

def package_collision_diagram(collision_diagram):
    if isinstance(collision_diagram, set):
        collision_diagram = list(collision_diagram)
        collision_diagram.sort()
        collision_diagram = [list(pair) for pair in collision_diagram]

    return collision_diagram

def package_data(N, seed, speed_data, survivors, still_alive, collision_diagram=None):
    collision_diagram = package_collision_diagram(collision_diagram=collision_diagram)

    return {
        "seed": seed,
        "N": N,
        "speed_data": speed_data,
        "survivors": dictify_bullets(bullets=survivors), 
        "still_alive": dictify_bullets(bullets=still_alive),
        "collision_diagram": collision_diagram
    }

def save_data(data, prefix=''):
    with open(get_fname(N=data['N'], seed=data['seed'], prefix=prefix), "w") as outfile:
        json.dump(data, outfile)

TEST_PREFIX = '_test_'

def load_data(N, seed, prefix=''):
    with open(get_fname(N=N, seed=seed, prefix=prefix), "r") as openfile:
        data = json.load(openfile)

    return data

def get_jsons(N):
    jsons = []
    for fname in os.listdir(DATA_DIR):
        if fname.startswith(f'N-{N}_seed'):
            with open(f"{DATA_DIR}/{fname}", 'r') as openfile:
                jsons.append(json.load(openfile))
    
    return jsons

ALIVE = 'alive_speed'
NONTHREATENED = 'non_threatened_speed'
UNDOOMED_LB = 'undoomed_lb'
UNDOOMED_UB = 'undoomed_ub'
FRONT_SPEEDS = [ALIVE, NONTHREATENED, UNDOOMED_LB, UNDOOMED_UB]
def get_front_speed_cdfs(N, speed_types=[ALIVE, NONTHREATENED]):
    result_dict = {speed_type: [] for speed_type in speed_types}
    jsons = get_jsons(N=N)
    for fname in os.listdir(DATA_DIR):
        if fname.startswith(f'N-{N}_seed'):
            with open(f"{DATA_DIR}/{fname}", 'r') as openfile:
                json_obj = json.load(openfile)

                for result_type, results in result_dict.items():
                    results.append(json_obj[result_type])
                    
                assert json_obj[ALIVE] <= json_obj[NONTHREATENED]
    
    return result_dict

def plot_front_speed_cdfs(N, speed_types=[ALIVE, NONTHREATENED]):
    result_dict = get_front_speed_cdfs(N=N, speed_types=speed_types)

    for result_type, results in result_dict.items():
        plot_cdf(data=results, label=result_type)
    leg = plt.legend(loc='upper left')

    num_simulations = len(results)
    if num_simulations > int(1e4):
        num_simulations = "{:.1e}".format(num_simulations)
    plt.title(f'{num_simulations} simulations')
    plt.show()

def survivors_json_to_bullets(survivors):
    if len(survivors) == 0:
        return survivors
    
    s = survivors[0]
    assert isinstance(s, list) or isinstance(s, dict)
    if isinstance(s, list):
        assert len(s) == 2
        assert isinstance(s[0], int)
        assert isinstance(s[1], float)
        return [Bullet(index=s[0], speed=s[1]) for s in survivors]
    
    else:
        return [Bullet(index=s['index'], speed=s['speed']) for s in survivors]

def get_num_survivors_results(N, cutoff=1.0):
    jsons = get_jsons(N=N)
    num_survivors = {}
    for obj in jsons:
        survivors = survivors_json_to_bullets(survivors=obj['survivors'])
        if cutoff < 1.0:
            survivors = [s for s in survivors if s.index < cutoff * N]

        num_survivors[obj['seed']] = len(survivors)

    return num_survivors

def non_eliminated_speed(N, bullet):
    interceptor = Bullet(index=N, speed=1)
    kill_stats = get_kill_stats(bullet=interceptor, target_bullet=bullet)
    return kill_stats.kill_pos / (kill_stats.kill_time + 1)

def non_eliminated_speeds_from_data(data):
    survivors = survivors_json_to_bullets(survivors=data['survivors'])

    if len(survivors) == 0:
        return data[NONTHREATENED], data[NONTHREATENED]
    
    survivor = survivors[0]
    if survivor.speed < data[ALIVE]:
        return data[NONTHREATENED], data[NONTHREATENED]

    lb = non_eliminated_speed(N=data['N'], bullet=survivor)
    num_floating_around = len(data['survivors']) + len(data['still_alive'])
    ub = non_eliminated_speed(N=data['N'] + num_floating_around - 1, bullet=survivor)
    assert lb <= ub <= data[NONTHREATENED]
    return max(lb, data[ALIVE]), max(ub, data[ALIVE])

def save_test_data():
    experiment(N=int(1e3), seed=TEST_SEED, prefix=TEST_PREFIX, find_collisions=True, save=True)

TEST_SEED = 42
TEST_N = int(1e3)
def load_test_data(prefix=TEST_PREFIX):
    return load_data(N=TEST_N, seed=TEST_SEED, prefix=prefix)

def end_to_end_with_speeds(seed=TEST_SEED, speeds=None):
    N = TEST_N
    ## so filename doesn't use seed 42 after code interrupted
    np.random.seed()
    prefix = f'Ztest_{np.random.randint(int(1e9))}_'
    fname = f"{prefix}N-{N}_seed-{seed}.json"
    assert fname not in os.listdir(DATA_DIR)

    data, bullets = experiment(N=N, seed=seed, speeds=speeds, prefix=prefix, find_collisions=True, save=True, to_time=False)
    assert fname in os.listdir(DATA_DIR)
    
    sur_idxs = set(s['index'] for s in data['survivors'])
    still_alive_idxs = set(b['index'] for b in data['still_alive'])
    col_idxs = set(c[0] for c in data['collision_diagram']) | set(c[1] for c in data['collision_diagram'])

    assert set(range(N)) == sur_idxs | still_alive_idxs | col_idxs
    assert still_alive_idxs - col_idxs == sur_idxs
    
    speed_data = data['speed_data']
    assert speed_data[NONTHREATENED] >= speed_data[UNDOOMED_UB] >= speed_data[UNDOOMED_LB] >= speed_data[ALIVE]

    test_data = load_test_data()
    assert data['seed'] == seed
    data['seed'] = TEST_SEED
    assert data == test_data
    data = load_data(N=N, seed=seed, prefix=prefix)
    assert data['seed'] == seed
    data['seed'] = TEST_SEED
    assert data == test_data

def end_to_end_test():
    start = time.time()
    end_to_end_with_speeds(seed=TEST_SEED, speeds=None)
    N = TEST_N
    np.random.seed(TEST_SEED)
    speeds = np.random.rand(N)
    np.random.seed()
    seed = np.random.randint(1e9)
    end_to_end_with_speeds(seed=seed, speeds=speeds)

    print('bullet.py noice!', time.time() - start)

end_to_end_test()
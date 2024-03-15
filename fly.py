from __future__ import annotations
import numpy as np
from dataclasses import dataclass
import time
import os
from typing import Optional
from sortedcontainers import SortedList
from sortedcontainers import SortedDict
import math
import bullets as bul
import importlib
importlib.reload(bul)

@dataclass
class Bullet:
    index:int
    speed:float
    next:Optional[Bullet] = None
    previous:Optional[Bullet] = None
    collision_time:Optional[float] = None
    def __str__(self):
        next = 'None' if self.next is None else self.next.index
        previous = 'None' if self.previous is None else self.previous.index
        return f'index: {self.index}, speed: {round(self.speed, 4)}, next: {next}, col_time: {self.collision_time}, prev: {previous}'

    def __lt__(self, other):
         return self.index < other.index
    
    def as_data(self):
        data = dict(self.__dict__)
        del data['next']
        del data['previous']
        return data
    
    # assumes next is not None
    def set_collision_time(self):
        v_j = self.speed
        v_i = self.next.speed
        if v_j <= v_i:
            return None
        j = self.index
        i = self.next.index

        self.collision_time = (j * v_j - i * v_i) / (v_j-v_i)
        return self.collision_time

def schedule_collision(bullet, collision_schedule, current_schedule=None, current_index=None):
    if bullet.next is None:
        return 
    
    collision_time = bullet.set_collision_time()
    if collision_time is None:
        return
    collision_slot = math.ceil(collision_time)
    if collision_slot == current_index:
        current_schedule.add((collision_time, bullet))
    else:
        if collision_slot not in collision_schedule:
            collision_schedule[collision_slot] = {}
        collision_schedule[collision_slot][bullet.index] = (collision_time, bullet)

def remove_collision(bullet, collision_schedule, current_schedule, current_index):
    if bullet.collision_time is None:
        return
    
    slot = math.ceil(bullet.collision_time)
    if slot == current_index:
        current_schedule.remove((bullet.collision_time, bullet))
    else:
        del collision_schedule[slot][bullet.index]
    bullet.collision_time = None

def get_threat_speed(bullet, collision_time):
    x = bullet.speed * (collision_time - bullet.index)
    return x / (collision_time + 1)

def update_maxes(maxes, left, N, lb_interceptor, ub_interceptor, collision_time=None):
    if collision_time is None:
        nt_speed = left.speed
    else:
        nt_speed = get_threat_speed(bullet=left, collision_time=collision_time)
    if nt_speed <= maxes[bul.ALIVE]:
        return

    new_maxes = {speed_type: nt_speed for speed_type in bul.FRONT_SPEEDS}
    if collision_time is None or collision_time > N:
        new_maxes[bul.ALIVE] = get_threat_speed(bullet=left, collision_time=N)
        for interceptor in (lb_interceptor, ub_interceptor):
            interceptor.next = left
            interceptor.set_collision_time()
        if collision_time is None or collision_time > lb_interceptor.collision_time:
            new_maxes[bul.UNDOOMED_LB] = get_threat_speed(
                bullet=left, collision_time=lb_interceptor.collision_time)
            new_maxes[bul.UNDOOMED_UB] = get_threat_speed(
                bullet=left, collision_time=ub_interceptor.collision_time)

    for speed_type in bul.FRONT_SPEEDS:
        maxes[speed_type] = max(maxes[speed_type], new_maxes[speed_type])

def run_current_collision_schedule(N, index, collision_schedule, current_schedule, collisions, 
                                   maxes, lb_interceptor=None, ub_interceptor=None, back=None):
    current_schedule = SortedList(current_schedule.values())

    while len(current_schedule) > 0:
        t, right = current_schedule.pop(index=0)
        left = right.next
        if collisions is not None:
            collisions.add((left.index, right.index))

        remove_collision(bullet=left, collision_schedule=collision_schedule, 
                         current_schedule=current_schedule, current_index=index)
        new_right = right.previous
        new_left = left.next
        if new_left is not None:
            new_left.previous = new_right
        else:
            update_maxes(maxes=maxes, N=N, left=left, collision_time=t, 
                         lb_interceptor=lb_interceptor, ub_interceptor=ub_interceptor)

        if new_right is not None:
            remove_collision(bullet=new_right, collision_schedule=collision_schedule, 
                         current_schedule=current_schedule, current_index=index)
            new_right.next = new_left
            schedule_collision(bullet=new_right, collision_schedule=collision_schedule, 
                               current_schedule=current_schedule, current_index=index)
        else:
            back = new_left
        
        del right
        del left

    return back

def listify_bullets(back):
    l = []
    bullet = back
    while bullet is not None:
        l.append(bullet.as_data())
        bullet = bullet.next
    l.reverse()
    return l

MILLION = int(1e6)
def run_bullet_process(N, speeds=None, track_collisions=False, to_time=True):
    start = time.time()
    milestone = start
    if speeds is None:
        speeds = np.random.rand(N)
    collision_schedule = {}
    collisions = set() if track_collisions else None
    maxes = bul.init_maxes()

    back = None
    for index in range(N):
        if to_time and (index + 1) % MILLION == 0:
            print("{:.1e}".format(index + 1), time.time() - milestone, time.time() - start)
            milestone = time.time()
        back = Bullet(index=index, speed=speeds[index], next=back)
        if back.next is not None:
            back.next.previous = back

        if index in collision_schedule:
            current_schedule = collision_schedule[index]
            del collision_schedule[index]
            run_current_collision_schedule(N=N, index=index, collision_schedule=collision_schedule, 
                current_schedule=current_schedule, collisions=collisions, maxes=maxes)
        schedule_collision(bullet=back, collision_schedule=collision_schedule)

    still_alive = listify_bullets(back=back)
    end_collision_schedule = SortedDict(collision_schedule)
    lb_interceptor = Bullet(index=N, speed=1)
    ub_interceptor = Bullet(index=(N + len(still_alive) - 1), speed=1)
    maxes = {speed_type: maxes[bul.ALIVE] for speed_type in maxes}
    while len(end_collision_schedule) > 0:
        index, current_schedule = end_collision_schedule.popitem(index=0)
        back = run_current_collision_schedule(N=N, index=index, collision_schedule=end_collision_schedule, 
            current_schedule=current_schedule, collisions=collisions, maxes=maxes, 
            lb_interceptor=lb_interceptor, ub_interceptor=ub_interceptor, back=back)
        
    survivors = listify_bullets(back=back)
    if len(survivors) > 0:
        front_dict = survivors[0]
        front = Bullet(index=front_dict['index'], speed=front_dict['speed'])
        update_maxes(maxes=maxes, left=front, N=N, lb_interceptor=lb_interceptor, 
                    ub_interceptor=ub_interceptor, collision_time=None)

    assert maxes[bul.ALIVE] <= maxes[bul.UNDOOMED_LB] <= maxes[bul.UNDOOMED_UB] <= maxes[bul.NONTHREATENED]
    if to_time:
        print('after N', time.time() - milestone, time.time() - start)
        
    return still_alive, maxes, survivors, collision_schedule, collisions

def package_data(N, seed, speed_data, survivors, still_alive, collision_diagram=None):
    collision_diagram = bul.package_collision_diagram(collision_diagram=collision_diagram)

    return {
        "seed": seed,
        "N": N,
        "speed_data": speed_data,
        "survivors": survivors, 
        "still_alive": still_alive,
        "collision_diagram": collision_diagram
    }

FLY_TEST_PREFIX = '_test_fly_'
def save_test_data():
    experiment(N=int(1e3), seed=bul.TEST_SEED, prefix=FLY_TEST_PREFIX, find_collisions=True, save=True)

def experiment(N, seed=None, speeds=None, prefix='', find_collisions=False, save=True, to_time=True):
    if seed is None:
        seed = np.random.randint(1e9)
    np.random.seed(seed)
    still_alive, speed_data, survivors, collision_schedule, collisions = run_bullet_process(
        N=N, speeds=speeds, track_collisions=find_collisions, to_time=to_time)
    data = package_data(N=N, seed=seed, speed_data=speed_data, survivors=survivors, 
                        still_alive=still_alive, collision_diagram=collisions)
    if save:
        bul.save_data(data=data, prefix=prefix)

    return data

def end_to_end_with_speeds(seed=bul.TEST_SEED, speeds=None):
    seed = bul.TEST_SEED
    N = bul.TEST_N
    ## so filename doesn't use seed 42 after code interrupted
    np.random.seed()
    prefix = f'Ztest_{np.random.randint(int(1e9))}_'
    fname = f"{prefix}N-{N}_seed-{seed}.json"
    assert fname not in os.listdir(bul.DATA_DIR)

    data = experiment(N=N, seed=seed, speeds=speeds, prefix=prefix, find_collisions=True, save=True, to_time=False)
    assert fname in os.listdir(bul.DATA_DIR)
    
    sur_idxs = set(s['index'] for s in data['survivors'])
    still_alive_idxs = set(b['index'] for b in data['still_alive'])
    col_idxs = set(c[0] for c in data['collision_diagram']) | set(c[1] for c in data['collision_diagram'])

    assert set(range(N)) == sur_idxs | still_alive_idxs | col_idxs
    assert still_alive_idxs - col_idxs == sur_idxs

    for survivor in data['survivors']:
        assert survivor['collision_time'] is None
    
    speed_data = data['speed_data']
    assert speed_data[bul.NONTHREATENED] >= speed_data[bul.UNDOOMED_UB] >= speed_data[bul.UNDOOMED_LB] >= speed_data[bul.ALIVE]

    test_data = bul.load_test_data(prefix=FLY_TEST_PREFIX)
    assert data['seed'] == seed
    data['seed'] = bul.TEST_SEED

    assert data == test_data
    data = bul.load_data(N=N, seed=seed, prefix=prefix)
    assert data['seed'] == seed
    data['seed'] = bul.TEST_SEED
    assert data == test_data

def end_to_end_test():
    start = time.time()
    end_to_end_with_speeds(speeds=None)
    N = bul.TEST_N
    np.random.seed(bul.TEST_SEED)
    speeds = np.random.rand(N)
    end_to_end_with_speeds(speeds=speeds)

    print('fly.py noice!!!', time.time() - start)
    
# save_test_data()
end_to_end_test()
# run_bullet_process(N=int(1e8))
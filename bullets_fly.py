from __future__ import annotations
import numpy as np
from dataclasses import dataclass
import time
from typing import Optional
from sortedcontainers import SortedList
from collections import defaultdict 
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
        return f'index: {self.index}, speed: {round(self.speed, 2)}, next: {next}, col_time: {self.collision_time}, prev: {previous}'

    def __lt__(self, other):
         return self.index < other.index
    
    def as_data(self):
        data = dict(self.__dict__)
        del data['next']
        del data['previous']
        return data

def schedule_collision(bullet, collision_schedule, current_schedule=None, current_index=None):
    if bullet.next is None:
        return 

    v_j = bullet.speed
    v_i = bullet.next.speed

    if v_j <= v_i:
        return 

    j = bullet.index
    i = bullet.next.index

    collision_time = (j * v_j - i * v_i)/(v_j-v_i)
    bullet.collision_time = collision_time
    collision_slot = math.ceil(collision_time)
    if collision_slot == current_index:
        current_schedule.add((bullet.collision_time, bullet))
    else:
        collision_schedule[collision_slot][bullet.index] = (bullet.collision_time, bullet)

def remove_collision(bullet, collision_schedule, current_schedule, current_index):
    if bullet.collision_time is None:
        return
    
    slot = math.ceil(bullet.collision_time)
    if slot == current_index:
        current_schedule.remove((bullet.collision_time, bullet))
    else:
        del collision_schedule[slot][bullet.index]
    bullet.collision_time = None

def run_current_collision_schedule(N, index, collision_schedule, collisions, maxes):
    if index not in collision_schedule:
        return
    
    current_schedule = SortedList(collision_schedule[index].values())
    del collision_schedule[index]

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
            if t < N:
                x = left.speed * (t - left.index)
                new_alive_speed = x / (t + 1)
                if new_alive_speed > maxes[bul.ALIVE]:
                    maxes[bul.ALIVE] = new_alive_speed
        if new_right is not None:
            remove_collision(bullet=new_right, collision_schedule=collision_schedule, 
                         current_schedule=current_schedule, current_index=index)
            new_right.next = new_left
            schedule_collision(bullet=new_right, collision_schedule=collision_schedule, 
                               current_schedule=current_schedule, current_index=index)
        
        del right
        del left

MILLION = int(1e6)
def run_bullet_process(N, track_collisions=False, to_time=True):
    start = time.time()
    milestone = start
    speeds = np.random.rand(N)
    collision_schedule = defaultdict(dict)
    collisions = set() if track_collisions else None
    maxes = bul.init_maxes()

    back = None
    for index in range(N):
        if (index + 1) % MILLION == 0:
            print("{:.1e}".format(index + 1), time.time() - milestone, time.time() - start)
            milestone = time.time()
        back = Bullet(index=index, speed=speeds[index], next=back)
        if back.next is not None:
            back.next.previous = back

        run_current_collision_schedule(N=N, index=index, collision_schedule=collision_schedule, 
                                       collisions=collisions, maxes=maxes)
        if index < N:
            schedule_collision(bullet=back, collision_schedule=collision_schedule)

    still_alive = []
    bullet = back
    while bullet is not None:
        still_alive.append(bullet.as_data())
        bullet = bullet.next
    still_alive.reverse()

    

    if to_time:
        print('run_bullet_process: (hh:mm:ss.ms) {}'.format(time.time() - start))
        
    return back, still_alive, maxes, collisions

def end_to_end_test():
    seed = 42
    N = int(1e3)
    ## so filename doesn't use seed 42 after code interrupted
    # np.random.seed()
    # prefix = f'Ztest_{np.random.randint(int(1e9))}_'
    # fname = f"{prefix}N-{N}_seed-{seed}.json"
    # assert fname not in os.listdir(DATA_DIR)
    np.random.seed(seed)
    back, still_alive, maxes, collisions = run_bullet_process(N=N, track_collisions=True, to_time=True)

    test_data = bul.load_test_data()
    test_collisions = set((col[0], col[1]) for col in test_data['collision_diagram'])
    assert collisions.issubset(test_collisions)
    still_alive_set = set((b['index'], b['speed']) for b in still_alive)
    test_still_alive_set = set((b['index'], b['speed']) for b in test_data['still_alive'])
    assert still_alive_set == test_still_alive_set
    print(test_data['speed_data'][bul.ALIVE])
    print(maxes[bul.ALIVE])

    # assert data == test_data
    # data = load_data(N=N, seed=seed, prefix=prefix)
    # assert data == test_data
    print('fly fly still noice!')

end_to_end_test()
# run_bullet_process(N=int(1e8))
from typing import List
import numpy as np
from utils import Particle

### 可以在这里写下一些你需要的变量和函数 ###
COLLISION_DISTANCE = 0.5
MAX_ERROR = 500
GAUSS_CHOICE = 0.008

def COLLISION(x, y, walls):
    for wall in walls:
        if abs(x-wall[0])<COLLISION_DISTANCE and abs(y-wall[1])<COLLISION_DISTANCE:
            return True
    return False

### 可以在这里写下一些你需要的变量和函数 ###


def generate_uniform_particles(walls, N):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    N: int, 采样点数量
    输出：
    particles: List[Particle], 返回在空地上均匀采样出的N个采样点的列表，每个点的权重都是1/N
    """
    all_particles: List[Particle] = []
    min_x,max_x = np.min(walls[:,0])+0.5,np.max(walls[:,0])-0.5
    min_y,max_y = np.min(walls[:,1])+0.5,np.max(walls[:,1])-0.5
    while len(all_particles) != N:
        x = np.random.uniform(min_x,max_x)
        y = np.random.uniform(min_y,max_y)
        theta = np.random.uniform(-np.pi,np.pi)
        if not COLLISION(x,y, walls):
            all_particles.append(Particle(x,y,theta,1/N))
            
    return all_particles


def calculate_particle_weight(estimated, gt):
    """
    输入：
    estimated: np.array, 该采样点的距离传感器数据
    gt: np.array, Pacman实际位置的距离传感器数据
    输出：
    weight, float, 该采样点的权重
    """
    weight, k = 1.0, 1.5
    error = np.linalg.norm(estimated-gt)
    if error < MAX_ERROR:
        weight =  np.exp( -k * error)
    else:
        weight = 0
    return weight


def resample_particles(walls, particles: List[Particle]):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    particles: List[Particle], 上一次采样得到的粒子，注意是按权重从大到小排列的
    输出：  
    particles: List[Particle], 返回重采样后的N个采样点的列表
    """
    resampled_particles: List[Particle] = []
    N = len(particles)
    min_x,max_x = np.min(walls[:,0])+0.5,np.max(walls[:,0])-0.5
    min_y,max_y = np.min(walls[:,1])+0.5,np.max(walls[:,1])-0.5
    sample_number = N
    sample_number2 = sample_number
    for particle in particles:
        number = int(sample_number * particle.weight+0.2)
        dx = np.random.normal(0, number*GAUSS_CHOICE, number)
        dy = np.random.normal(0, number*GAUSS_CHOICE, number)
        dtheta = np.random.normal(0, number*GAUSS_CHOICE/2, number)
        sample_number2 -= number
        for i in range(0, number):
            random_x = particle.position[0] + dx[i]
            random_x = np.clip(random_x,min_x,max_x)
            random_y = particle.position[1] + dy[i]
            random_y = np.clip(random_y,min_y,max_y)
            theta = particle.theta + dtheta[i]
            while theta > np.pi: theta -= np.pi
            while theta < -np.pi: theta += np.pi
            resampled_particles.append(Particle(random_x,random_y,theta,1))
    random_sample_particles = generate_uniform_particles(walls, sample_number2)
    for particle in random_sample_particles:
        particle.weight = 1
    resampled_particles.extend(random_sample_particles)
    print(len(resampled_particles))
    return resampled_particles

def apply_state_transition(p: Particle, traveled_distance, dtheta):
    """
    输入：
    p: 采样的粒子
    traveled_distance, dtheta: ground truth的Pacman这一步相对于上一步运动方向改变了dtheta，并移动了traveled_distance的距离
    particle: 按照相同方式进行移动后的粒子
    """
    p.theta = (p.theta + dtheta)
    while p.theta > np.pi: p.theta -= np.pi
    while p.theta < -np.pi: p.theta += np.pi
    p.position[0] += traveled_distance * np.cos(p.theta)
    p.position[1] += traveled_distance * np.sin(p.theta)
    return p

def get_estimate_result(particles: List[Particle]):
    """
    输入：
    particles: List[Particle], 全部采样粒子
    输出：
    final_result: Particle, 最终的猜测结果
    """
    return particles[0]
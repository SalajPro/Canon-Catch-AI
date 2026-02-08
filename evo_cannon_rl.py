import os
import sys
import time
import math
import json
import csv
import random
import signal
import pickle
import threading
from dataclasses import dataclass
from io import BytesIO

import numpy as np

import tkinter as tk
from tkinter import ttk, messagebox
import multiprocessing as mp


BALL_THROW_DELAY = 1.4    
BASKET_SPEED = 1000.0      
G = 850.0                 

THROWS_PER_EP = 10
MAX_MISSES = 5

POP_SIZE = 10
KILL_COUNT = 5             

WIDTH, HEIGHT = 1000, 600
GROUND_Y = int(HEIGHT * 0.85)

DT = 1.0 / 60.0           
DECISION_EVERY_N_FRAMES = 4  

BALL_RADII = [12, 14, 16]
BALL_SPEED_MIN = 650.0
BALL_SPEED_MAX = 1050.0

BOUNCE_WALLS = True
BOUNCE_DAMP_X = 0.85
BOUNCE_DAMP_Y = 0.75

CANNON_X = WIDTH // 2
CANNON_Y = 85

ANGLE_DEG_MIN = 205
ANGLE_DEG_MAX = 335

RANDOMIZE_THROW = True
DELAY_RANDOM_MIN = 0.7
DELAY_RANDOM_MAX = 1.3

FIT_W = 10.0
FIT_L = 6.0
FIT_S = 2.0

RUN_DIR = "runs_evo_cannon"
CKPT_DIR = os.path.join(RUN_DIR, "checkpoints")
BEST_PATH = os.path.join(RUN_DIR, "best_model.npz")
BEST_META_PATH = os.path.join(RUN_DIR, "best_meta.json")
LOG_PATH = os.path.join(RUN_DIR, "training_log.csv")


def tanh(x):
    return np.tanh(x)

@dataclass
class NetParams:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    W3: np.ndarray
    b3: np.ndarray

def init_net(rng: np.random.Generator, in_dim=5, h1=16, h2=16, out_dim=3) -> NetParams:
    def w(shape):
        fan_in = shape[0]
        scale = 1.0 / math.sqrt(max(1, fan_in))
        return rng.normal(0, scale, size=shape).astype(np.float32)

    return NetParams(
        W1=w((in_dim, h1)), b1=np.zeros((h1,), np.float32),
        W2=w((h1, h2)), b2=np.zeros((h2,), np.float32),
        W3=w((h2, out_dim)), b3=np.zeros((out_dim,), np.float32),
    )

def forward(net: NetParams, obs: np.ndarray) -> int:
    x = obs.astype(np.float32)
    x = tanh(x @ net.W1 + net.b1)
    x = tanh(x @ net.W2 + net.b2)
    logits = x @ net.W3 + net.b3
    return int(np.argmax(logits))

def clone_net(net: NetParams) -> NetParams:
    return NetParams(
        W1=net.W1.copy(), b1=net.b1.copy(),
        W2=net.W2.copy(), b2=net.b2.copy(),
        W3=net.W3.copy(), b3=net.b3.copy(),
    )

def mutate_net(rng: np.random.Generator, net: NetParams, mutation_rate: float, mutation_sigma: float) -> NetParams:
    child = clone_net(net)

    def mutate_array(a: np.ndarray):
        mask = rng.random(a.shape) < mutation_rate
        noise = rng.normal(0, mutation_sigma, size=a.shape).astype(np.float32)
        a[mask] += noise[mask]

    mutate_array(child.W1); mutate_array(child.b1)
    mutate_array(child.W2); mutate_array(child.b2)
    mutate_array(child.W3); mutate_array(child.b3)
    return child

def net_to_bytes(net: NetParams) -> bytes:
    payload = {
        "W1": net.W1, "b1": net.b1,
        "W2": net.W2, "b2": net.b2,
        "W3": net.W3, "b3": net.b3,
    }
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

def bytes_to_net(b: bytes) -> NetParams:
    d = pickle.loads(b)
    return NetParams(d["W1"], d["b1"], d["W2"], d["b2"], d["W3"], d["b3"])

def save_best_npz(path: str, net: NetParams):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, W1=net.W1, b1=net.b1, W2=net.W2, b2=net.b2, W3=net.W3, b3=net.b3)

def load_best_npz(path: str) -> NetParams:
    data = np.load(path, allow_pickle=False)
    return NetParams(
        data["W1"].astype(np.float32),
        data["b1"].astype(np.float32),
        data["W2"].astype(np.float32),
        data["b2"].astype(np.float32),
        data["W3"].astype(np.float32),
        data["b3"].astype(np.float32),
    )

@dataclass
class Ball:
    x: float
    y: float
    vx: float
    vy: float
    r: int

def compute_next_spawn_delay(rng: np.random.Generator) -> float:
    if RANDOMIZE_THROW:
        return float(rng.uniform(BALL_THROW_DELAY * DELAY_RANDOM_MIN, BALL_THROW_DELAY * DELAY_RANDOM_MAX))
    return float(BALL_THROW_DELAY)

def spawn_ball(rng: np.random.Generator) -> Ball:
    angle_deg = rng.uniform(ANGLE_DEG_MIN, ANGLE_DEG_MAX)
    angle = math.radians(angle_deg)
    speed = rng.uniform(BALL_SPEED_MIN, BALL_SPEED_MAX)

    vx = math.cos(angle) * speed
    vy = math.sin(angle) * speed  

    start_x = CANNON_X + math.cos(angle) * 80.0
    start_y = CANNON_Y + math.sin(angle) * 80.0
    r = int(rng.choice(BALL_RADII))
    return Ball(x=float(start_x), y=float(start_y), vx=float(vx), vy=float(vy), r=r)

def step_ball(ball: Ball):
    ball.vy += G * DT
    ball.x += ball.vx * DT
    ball.y += ball.vy * DT

    if BOUNCE_WALLS:
        if ball.x - ball.r <= 0:
            ball.x = ball.r
            ball.vx *= -BOUNCE_DAMP_X
        elif ball.x + ball.r >= WIDTH:
            ball.x = WIDTH - ball.r
            ball.vx *= -BOUNCE_DAMP_X

        if ball.y - ball.r <= 0:
            ball.y = ball.r
            ball.vy *= -BOUNCE_DAMP_Y

def ball_hit_ground(ball: Ball) -> bool:
    return (ball.y + ball.r) >= GROUND_Y

def circle_rect_collide(cx, cy, r, rect):
    rx, ry, rw, rh = rect
    closest_x = clamp(cx, rx, rx + rw)
    closest_y = clamp(cy, ry, ry + rh)
    dx = cx - closest_x
    dy = cy - closest_y
    return (dx*dx + dy*dy) <= (r*r)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def basket_rect(basket_x: float):
    w, h = 140, 62
    y = GROUND_Y - 10
    rx = basket_x - w/2
    ry = y - h
    return (rx, ry, w, h)

def obs_from_state(basket_x: float, ball: Ball) -> np.ndarray:
    bx = basket_x / WIDTH
    x = ball.x / WIDTH
    y = ball.y / HEIGHT
    vx = ball.vx / 1200.0
    vy = ball.vy / 1200.0
    return np.array([bx, x, y, vx, vy], dtype=np.float32)

@dataclass
class EpisodeResult:
    fitness: float
    W: int
    L: int
    S: int

def run_episode_for_agent(rng: np.random.Generator, net: NetParams) -> EpisodeResult:
    basket_x = WIDTH / 2.0
    W = 0
    L = 0
    streak = 0
    best_streak = 0

    throws_done = 0
    frames = 0

    spawn_timer = 0.0
    next_spawn = compute_next_spawn_delay(rng)
    ball = None

    while throws_done < THROWS_PER_EP and L < MAX_MISSES:
        if frames % DECISION_EVERY_N_FRAMES == 0:
            if ball is not None:
                obs = obs_from_state(basket_x, ball)
                action = forward(net, obs) 
            else:
                action = 1

            move_dir = (-1 if action == 0 else (1 if action == 2 else 0))
            basket_x += move_dir * BASKET_SPEED * (DT * DECISION_EVERY_N_FRAMES)
            basket_x = clamp(basket_x, 140/2 + 10, WIDTH - 140/2 - 10)

        if ball is None:
            spawn_timer += DT
            if spawn_timer >= next_spawn:
                spawn_timer = 0.0
                next_spawn = compute_next_spawn_delay(rng)
                ball = spawn_ball(rng)

        if ball is not None:
            step_ball(ball)

            rect = basket_rect(basket_x)
            caught = circle_rect_collide(ball.x, ball.y, ball.r, rect) and (ball.vy > 0)

            if caught:
                W += 1
                streak += 1
                best_streak = max(best_streak, streak)
                throws_done += 1
                ball = None 
            else:
                if ball_hit_ground(ball):
                    L += 1
                    streak = 0
                    throws_done += 1
                    ball = None

        frames += 1
        if frames > 60 * 45:  
            break

    fitness = (W * FIT_W) - (L * FIT_L) + (best_streak * FIT_S)
    return EpisodeResult(fitness=float(fitness), W=W, L=L, S=best_streak)


@dataclass
class AgentStat:
    fitness: float = -1e9
    W: int = 0
    L: int = 0
    S: int = 0
    parent: str = ""
    is_elite: bool = False

def ensure_dirs():
    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

def init_log_csv():
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["gen", "best_fitness", "best_W", "best_L", "best_S",
                        "avg_fitness", "mutation_rate", "mutation_sigma", "best_model_path"])

def append_log(gen, best_res: EpisodeResult, avg_fit, mrate, msigma, best_path):
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([gen, best_res.fitness, best_res.W, best_res.L, best_res.S,
                    avg_fit, mrate, msigma, best_path])

def save_meta(gen, best_res: EpisodeResult, mrate, msigma):
    meta = {
        "gen": gen,
        "best_fitness": best_res.fitness,
        "best_W": best_res.W,
        "best_L": best_res.L,
        "best_S": best_res.S,
        "mutation_rate": mrate,
        "mutation_sigma": msigma,
        "BALL_THROW_DELAY": BALL_THROW_DELAY,
        "BASKET_SPEED": BASKET_SPEED,
        "G": G,
        "THROWS_PER_EP": THROWS_PER_EP,
        "MAX_MISSES": MAX_MISSES,
    }
    with open(BEST_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def adaptive_mutation_update(best_fitness, prev_best_fitness, no_improve_count, mutation_rate, mutation_sigma):
    if best_fitness > prev_best_fitness + 1e-6:
        no_improve_count = 0
        mutation_rate *= 0.92
        mutation_sigma *= 0.92
    else:
        no_improve_count += 1
        if no_improve_count >= 10:
            mutation_rate *= 1.08
            mutation_sigma *= 1.08

    mutation_rate = float(clamp(mutation_rate, 0.02, 0.35))
    mutation_sigma = float(clamp(mutation_sigma, 0.01, 0.25))
    return mutation_rate, mutation_sigma, no_improve_count

def weighted_parent_choice(rng: np.random.Generator, elite_indices):
    weights = np.array([5, 4, 3, 2, 1], dtype=np.float32)
    weights = weights / weights.sum()
    idx = int(rng.choice(elite_indices, p=weights))
    return idx


def viewer_process(shared, stop_event: mp.Event):
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Best Agent (Live)")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("consolas", 22)

    def draw_vertical_gradient(surf, rect, top, bottom):
        x, y, w, h = rect
        for i in range(h):
            t = i / max(1, h - 1)
            c = (
                int(top[0] + (bottom[0] - top[0]) * t),
                int(top[1] + (bottom[1] - top[1]) * t),
                int(top[2] + (bottom[2] - top[2]) * t),
            )
            pygame.draw.line(surf, c, (x, y + i), (x + w, y + i))

    def draw_background():
        draw_vertical_gradient(screen, (0, 0, WIDTH, HEIGHT), (130, 205, 255), (18, 55, 130))
        for r, a in [(95, 25), (70, 35), (45, 60)]:
            glow = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow, (255, 240, 190, a), (r, r), r)
            screen.blit(glow, (WIDTH - 180, 40))
        pygame.draw.circle(screen, (255, 245, 200), (WIDTH - 140, 95), 34)

        # clouds
        def cloud(cx, cy, scale=1.0):
            parts = [(0, 0, 42), (35, -10, 34), (65, 0, 40), (30, 12, 38)]
            for ox, oy, rr in parts:
                pygame.draw.circle(screen, (245, 250, 255),
                                   (int(cx + ox * scale), int(cy + oy * scale)), int(rr * scale))
            pygame.draw.ellipse(screen, (235, 242, 250), (cx - 10, cy + 10, int(110 * scale), int(35 * scale)))

        cloud(180, 120, 0.9)
        cloud(420, 80, 0.8)
        cloud(650, 140, 1.0)

        draw_vertical_gradient(screen, (0, GROUND_Y, WIDTH, HEIGHT - GROUND_Y),
                               (85, 195, 105), (35, 125, 70))
        for x in range(0, WIDTH, 55):
            pygame.draw.line(screen, (30, 105, 65), (x, GROUND_Y + 20), (x + 25, HEIGHT), 2)

    def draw_basket(basket_x):
        w, h = 140, 62
        y = GROUND_Y - 10
        rx = int(basket_x - w/2)
        ry = int(y - h)
        sh = pygame.Surface((w + 40, 18), pygame.SRCALPHA)
        pygame.draw.ellipse(sh, (0, 0, 0, 80), sh.get_rect())
        screen.blit(sh, (rx - 20, GROUND_Y - 8))

        body = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(body, (185, 125, 70), (0, 6, w, h - 6), border_radius=16)
        pygame.draw.rect(body, (140, 90, 50), (0, 18, w, h - 18), border_radius=16)
        pygame.draw.rect(body, (230, 180, 120), (10, 6, w - 20, 12), border_radius=10)
        for i in range(10, w - 10, 14):
            pygame.draw.line(body, (115, 70, 38), (i, 22), (i, h - 8), 2)
        pygame.draw.rect(body, (60, 35, 20), (0, 6, w, h - 6), 3, border_radius=16)

        screen.blit(body, (rx, ry))

    def draw_ball(ball: Ball):
        dist = clamp((GROUND_Y - ball.y) / 350.0, 0.0, 1.0)
        sw = int(ball.r * (2.4 - 1.2 * dist))
        sh = int(ball.r * (0.85 - 0.4 * dist))
        shadow = pygame.Surface((sw * 2, sh * 2), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0, 0, 0, 85), shadow.get_rect())
        screen.blit(shadow, (ball.x - shadow.get_width() // 2, GROUND_Y - sh))

        pygame.draw.circle(screen, (230, 70, 70), (int(ball.x), int(ball.y)), ball.r)
        pygame.draw.circle(screen, (255, 155, 155),
                           (int(ball.x - ball.r * 0.35), int(ball.y - ball.r * 0.35)),
                           int(ball.r * 0.35))
        pygame.draw.circle(screen, (120, 25, 25), (int(ball.x), int(ball.y)), ball.r, 3)

    def draw_cannon(angle_rad):
        cx, cy = CANNON_X, CANNON_Y
        pygame.draw.circle(screen, (45, 45, 55), (cx, cy), 26)
        pygame.draw.circle(screen, (25, 25, 30), (cx, cy), 26, 4)

        length = 95
        thickness = 18
        barrel = pygame.Surface((length + 40, thickness + 40), pygame.SRCALPHA)
        pygame.draw.rect(barrel, (95, 95, 110), (20, 20, length, thickness), border_radius=12)
        pygame.draw.rect(barrel, (40, 40, 50), (20, 20, length, thickness), 4, border_radius=12)

        rotated = pygame.transform.rotate(barrel, -math.degrees(angle_rad))
        rect = rotated.get_rect(center=(cx, cy))
        screen.blit(rotated, rect.topleft)

        mx = cx + math.cos(angle_rad) * length
        my = cy + math.sin(angle_rad) * length
        glow = pygame.Surface((60, 60), pygame.SRCALPHA)
        pygame.draw.circle(glow, (255, 230, 180, 90), (30, 30), 18)
        screen.blit(glow, (mx - 30, my - 30))

    rng = np.random.default_rng(int(time.time()) & 0xFFFFFFFF)

    net = init_net(rng)

    basket_x = WIDTH / 2.0
    ball = None
    spawn_timer = 0.0
    next_spawn = compute_next_spawn_delay(rng)
    frames = 0

    # Episode stats for W/L/S in viewer
    W = 0
    L = 0
    streak = 0
    best_streak = 0
    throws_done = 0

    last_angle = math.radians(270)

    while not stop_event.is_set():
        # Handle window events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop_event.set()

        try:
            b = shared.get("best_net_bytes", None)
            if b is not None:
                net = bytes_to_net(b)
        except Exception:
            pass

        # Reset episode when done
        if throws_done >= THROWS_PER_EP or L >= MAX_MISSES:
            W = L = 0
            streak = 0
            best_streak = 0
            throws_done = 0
            basket_x = WIDTH / 2.0
            ball = None
            spawn_timer = 0.0
            next_spawn = compute_next_spawn_delay(rng)
            frames = 0

        if frames % DECISION_EVERY_N_FRAMES == 0 and ball is not None:
            obs = obs_from_state(basket_x, ball)
            action = forward(net, obs)
            move_dir = (-1 if action == 0 else (1 if action == 2 else 0))
            basket_x += move_dir * BASKET_SPEED * (DT * DECISION_EVERY_N_FRAMES)
            basket_x = clamp(basket_x, 140/2 + 10, WIDTH - 140/2 - 10)

        if ball is None:
            spawn_timer += DT
            if spawn_timer >= next_spawn:
                spawn_timer = 0.0
                next_spawn = compute_next_spawn_delay(rng)
                ball = spawn_ball(rng)
                last_angle = math.atan2(ball.vy, ball.vx)

        if ball is not None:
            step_ball(ball)
            rect = basket_rect(basket_x)
            caught = circle_rect_collide(ball.x, ball.y, ball.r, rect) and (ball.vy > 0)
            if caught:
                W += 1
                streak += 1
                best_streak = max(best_streak, streak)
                throws_done += 1
                ball = None
            else:
                if ball_hit_ground(ball):
                    L += 1
                    streak = 0
                    throws_done += 1
                    ball = None

        draw_background()

        tsec = pygame.time.get_ticks() / 1000.0
        wiggle = math.sin(tsec * 1.5) * math.radians(10)
        draw_cannon(last_angle + wiggle)

        if ball is not None:
            draw_ball(ball)
        draw_basket(basket_x)

        gen = shared.get("gen", 0)
        best_fit = shared.get("best_fitness", 0.0)
        mrate = shared.get("mutation_rate", 0.0)
        msig = shared.get("mutation_sigma", 0.0)

        hud1 = font.render(f"GEN: {gen}  BestFit: {best_fit:.2f}  mRate: {mrate:.3f}  mSig: {msig:.3f}", True, (255, 255, 255))
        hud2 = font.render(f"Viewer Episode: Throws {throws_done}/{THROWS_PER_EP}   W/L/S = {W}/{L}/{best_streak}", True, (235, 240, 250))
        screen.blit(hud1, (18, 16))
        screen.blit(hud2, (18, 44))

        pygame.display.flip()
        clock.tick(60)
        frames += 1

    pygame.quit()


def training_process(shared, stop_event: mp.Event):
    ensure_dirs()
    init_log_csv()

    seed = int(time.time()) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)

    population = []
    stats = [AgentStat() for _ in range(POP_SIZE)]

    if os.path.exists(BEST_PATH):
        try:
            best = load_best_npz(BEST_PATH)
            population.append(best)
            for _ in range(POP_SIZE - 1):
                population.append(mutate_net(rng, best, mutation_rate=0.10, mutation_sigma=0.06))
        except Exception:
            population = [init_net(rng) for _ in range(POP_SIZE)]
    else:
        population = [init_net(rng) for _ in range(POP_SIZE)]
    mutation_rate = 0.10
    mutation_sigma = 0.06
    no_improve = 0
    prev_best_fit = -1e9

    gen = int(shared.get("gen", 0))

    def save_best_everything(best_net: NetParams, best_res: EpisodeResult):
        save_best_npz(BEST_PATH, best_net)
        save_meta(gen, best_res, mutation_rate, mutation_sigma)
        # Also keep occasional checkpoint
        if gen % 20 == 0:
            cp = os.path.join(CKPT_DIR, f"best_gen_{gen:06d}.npz")
            save_best_npz(cp, best_net)

    def handle_sigint(signum, frame):
        stop_event.set()
    signal.signal(signal.SIGINT, handle_sigint)

    while not stop_event.is_set():
        gen += 1

        fits = []
        results = []
        for i in range(POP_SIZE):
            res = run_episode_for_agent(rng, population[i])
            results.append(res)
            fits.append(res.fitness)

            stats[i].fitness = res.fitness
            stats[i].W = res.W
            stats[i].L = res.L
            stats[i].S = res.S
            stats[i].is_elite = False

        avg_fit = float(np.mean(fits))

        order = list(range(POP_SIZE))
        order.sort(key=lambda i: stats[i].fitness, reverse=True)

        elite = order[:POP_SIZE - KILL_COUNT]  
        dead = order[POP_SIZE - KILL_COUNT:] 

        for rank, idx in enumerate(elite):
            stats[idx].is_elite = True
            stats[idx].parent = "survivor"

        best_idx = elite[0]
        best_res = results[best_idx]
        best_net = population[best_idx]

        mutation_rate, mutation_sigma, no_improve = adaptive_mutation_update(
            best_res.fitness, prev_best_fit, no_improve, mutation_rate, mutation_sigma
        )
        prev_best_fit = max(prev_best_fit, best_res.fitness)

        for di in dead:
            pidx = weighted_parent_choice(rng, elite)
            child = mutate_net(rng, population[pidx], mutation_rate, mutation_sigma)
            population[di] = child
            stats[di].parent = f"mut({pidx})"
            stats[di].is_elite = False

        try:
            save_best_everything(best_net, best_res)
        except Exception:
            pass

        append_log(gen, best_res, avg_fit, mutation_rate, mutation_sigma, BEST_PATH)

        try:
            shared["gen"] = gen
            shared["best_fitness"] = float(best_res.fitness)
            shared["mutation_rate"] = float(mutation_rate)
            shared["mutation_sigma"] = float(mutation_sigma)
            shared["best_W"] = int(best_res.W)
            shared["best_L"] = int(best_res.L)
            shared["best_S"] = int(best_res.S)
            shared["best_net_bytes"] = net_to_bytes(best_net)

            table = []
            for i in range(POP_SIZE):
                table.append({
                    "id": i,
                    "fitness": float(stats[i].fitness),
                    "W": int(stats[i].W),
                    "L": int(stats[i].L),
                    "S": int(stats[i].S),
                    "elite": bool(stats[i].is_elite),
                    "parent": stats[i].parent
                })
            shared["table"] = table
        except Exception:
            pass

        time.sleep(0.01)

    try:
        b = shared.get("best_net_bytes", None)
        if b is not None:
            bn = bytes_to_net(b)
            res = EpisodeResult(
                fitness=float(shared.get("best_fitness", 0.0)),
                W=int(shared.get("best_W", 0)),
                L=int(shared.get("best_L", 0)),
                S=int(shared.get("best_S", 0)),
            )
            save_best_npz(BEST_PATH, bn)
            save_meta(int(shared.get("gen", gen)), res, float(shared.get("mutation_rate", mutation_rate)), float(shared.get("mutation_sigma", mutation_sigma)))
    except Exception:
        pass

class App:
    def __init__(self, root):
        self.root = root
        root.title("Evo Cannon RL Control Panel")

        self.manager = mp.Manager()
        self.shared = self.manager.dict()
        self.stop_event = mp.Event()

        self.train_proc = None
        self.view_proc = None
        self.view_stop = mp.Event()

        self.status_var = tk.StringVar(value="Idle.")
        self.best_var = tk.StringVar(value="Best: -")
        self.mut_var = tk.StringVar(value="Mutation: -")
        self.ep_var = tk.StringVar(value=f"EP: {THROWS_PER_EP} throws / {MAX_MISSES} misses  |  Delay={BALL_THROW_DELAY}s  Speed={BASKET_SPEED}  G={G}")

        top = ttk.Frame(root, padding=10)
        top.pack(fill="x")

        ttk.Label(top, textvariable=self.ep_var).pack(anchor="w")
        ttk.Label(top, textvariable=self.status_var, font=("Consolas", 11)).pack(anchor="w", pady=(6, 0))
        ttk.Label(top, textvariable=self.best_var, font=("Consolas", 11)).pack(anchor="w")
        ttk.Label(top, textvariable=self.mut_var, font=("Consolas", 11)).pack(anchor="w")

        btns = ttk.Frame(root, padding=10)
        btns.pack(fill="x")

        self.start_btn = ttk.Button(btns, text="Start Training", command=self.start_training)
        self.stop_btn = ttk.Button(btns, text="Stop Training (Save Best)", command=self.stop_training)
        self.view_btn = ttk.Button(btns, text="View Best (Live)", command=self.start_viewer)
        self.close_view_btn = ttk.Button(btns, text="Close Viewer", command=self.stop_viewer)

        self.start_btn.pack(side="left", padx=5)
        self.stop_btn.pack(side="left", padx=5)
        self.view_btn.pack(side="left", padx=5)
        self.close_view_btn.pack(side="left", padx=5)

        table_frame = ttk.Frame(root, padding=10)
        table_frame.pack(fill="both", expand=True)

        cols = ("id", "fitness", "W", "L", "S", "elite", "parent")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=12)
        for c in cols:
            self.tree.heading(c, text=c.upper())
            self.tree.column(c, width=100 if c != "parent" else 140, anchor="center")
        self.tree.pack(fill="both", expand=True)

        for i in range(POP_SIZE):
            self.tree.insert("", "end", values=(i, "-", "-", "-", "-", "-", "-"))

        self.root.after(200, self.poll_updates)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def start_training(self):
        if self.train_proc is not None and self.train_proc.is_alive():
            messagebox.showinfo("Training", "Training already running.")
            return

        self.stop_event.clear()
        self.shared["gen"] = int(self.shared.get("gen", 0))
        self.shared["best_fitness"] = float(self.shared.get("best_fitness", 0.0))
        self.shared["mutation_rate"] = float(self.shared.get("mutation_rate", 0.10))
        self.shared["mutation_sigma"] = float(self.shared.get("mutation_sigma", 0.06))

        self.train_proc = mp.Process(target=training_process, args=(self.shared, self.stop_event), daemon=True)
        self.train_proc.start()
        self.status_var.set("Training started.")
        self.start_btn.state(["disabled"])

    def stop_training(self):
        if self.train_proc is None or not self.train_proc.is_alive():
            self.status_var.set("Not training.")
            self.start_btn.state(["!disabled"])
            return
        self.status_var.set("Stopping training (saving best)...")
        self.stop_event.set()
        self.root.after(200, self._join_training)

    def _join_training(self):
        if self.train_proc is not None and self.train_proc.is_alive():
            self.root.after(200, self._join_training)
            return
        self.status_var.set(f"Stopped. Best saved to {BEST_PATH}")
        self.start_btn.state(["!disabled"])

    def start_viewer(self):
        if self.view_proc is not None and self.view_proc.is_alive():
            self.status_var.set("Viewer already running.")
            return
        self.view_stop.clear()
        self.view_proc = mp.Process(target=viewer_process, args=(self.shared, self.view_stop), daemon=True)
        self.view_proc.start()
        self.status_var.set("Viewer started (live best).")

    def stop_viewer(self):
        if self.view_proc is None or not self.view_proc.is_alive():
            self.status_var.set("Viewer not running.")
            return
        self.view_stop.set()
        self.status_var.set("Closing viewer...")

    def poll_updates(self):
        try:
            gen = int(self.shared.get("gen", 0))
            best_fit = float(self.shared.get("best_fitness", 0.0))
            mrate = float(self.shared.get("mutation_rate", 0.0))
            msig = float(self.shared.get("mutation_sigma", 0.0))
            bw = int(self.shared.get("best_W", 0))
            bl = int(self.shared.get("best_L", 0))
            bs = int(self.shared.get("best_S", 0))

            self.best_var.set(f"Best: fitness={best_fit:.2f}  W/L/S={bw}/{bl}/{bs}")
            self.mut_var.set(f"Mutation: rate={mrate:.3f}  sigma={msig:.3f}  |  Gen={gen}")

            table = self.shared.get("table", None)
            if table is not None and isinstance(table, list) and len(table) == POP_SIZE:
                # Update rows
                items = self.tree.get_children()
                for row_idx, item in enumerate(items):
                    a = table[row_idx]
                    elite_mark = "X" if a.get("elite") else ""
                    self.tree.item(item, values=(
                        a.get("id"),
                        f"{a.get('fitness'):.2f}",
                        a.get("W"),
                        a.get("L"),
                        a.get("S"),
                        elite_mark,
                        a.get("parent"),
                    ))
        except Exception:
            pass

        if self.train_proc is not None and (not self.train_proc.is_alive()) and (not self.stop_event.is_set()):
            self.start_btn.state(["!disabled"])

        self.root.after(200, self.poll_updates)

    def on_close(self):
        try:
            if self.train_proc is not None and self.train_proc.is_alive():
                self.stop_event.set()
            if self.view_proc is not None and self.view_proc.is_alive():
                self.view_stop.set()
        except Exception:
            pass
        self.root.destroy()


def main():
    mp.set_start_method("spawn", force=True)

    ensure_dirs()
    init_log_csv()

    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass

    app = App(root)

    def on_sigint(signum, frame):
        try:
            app.stop_training()
        except Exception:
            pass

    signal.signal(signal.SIGINT, on_sigint)

    root.mainloop()


if __name__ == "__main__":
    main()

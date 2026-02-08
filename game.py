import pygame
import random
import math
import numpy as np 

BALL_THROW_DELAY = 1.4     
RANDOMIZE_THROW = True     
DELAY_RANDOM_MIN = 0.7     
DELAY_RANDOM_MAX = 1.3     

BOUNCE_WALLS = True        
BOUNCE_DAMP_X = 0.85     
BOUNCE_DAMP_Y = 0.75       

pygame.init()
WIDTH, HEIGHT = 1000, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Cannon Catch AI")
clock = pygame.time.Clock()

GROUND_Y = int(HEIGHT * 0.85)
GRAVITY = 850.0

WHITE = (255, 255, 255)
AI_MODEL_PATH = "runs_evo_cannon/best_model.npz"
AI_MODE = True 

class NetParams:
    def __init__(self, W1, b1, W2, b2, W3, b3):
        self.W1, self.b1 = W1, b1
        self.W2, self.b2 = W2, b2
        self.W3, self.b3 = W3, b3

def load_best_npz(path: str):
    d = np.load(path)
    return NetParams(
        d["W1"].astype(np.float32),
        d["b1"].astype(np.float32),
        d["W2"].astype(np.float32),
        d["b2"].astype(np.float32),
        d["W3"].astype(np.float32),
        d["b3"].astype(np.float32),
    )

def forward(net: NetParams, obs: np.ndarray) -> int:
    x = obs.astype(np.float32)
    x = np.tanh(x @ net.W1 + net.b1)
    x = np.tanh(x @ net.W2 + net.b2)
    logits = x @ net.W3 + net.b3
    return int(np.argmax(logits))

def make_obs(basket_x: float, ball) -> np.ndarray:
    return np.array([
        basket_x / WIDTH,
        ball.x / WIDTH,
        ball.y / HEIGHT,
        ball.vx / 1200.0,
        ball.vy / 1200.0
    ], dtype=np.float32)

AI_NET = None
try:
    AI_NET = load_best_npz(AI_MODEL_PATH)
except Exception as e:
    print(f"[AI] Could not load model at {AI_MODEL_PATH}: {e}")
    print("[AI] Starting in MANUAL mode. Put best_model.npz in runs_evo_cannon/ and restart.")
    AI_MODE = False

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

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
    # Sky
    draw_vertical_gradient(screen, (0, 0, WIDTH, HEIGHT), (130, 205, 255), (18, 55, 130))

    # Sun glow
    for r, a in [(95, 25), (70, 35), (45, 60)]:
        glow = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow, (255, 240, 190, a), (r, r), r)
        screen.blit(glow, (WIDTH - 180, 40))
    pygame.draw.circle(screen, (255, 245, 200), (WIDTH - 140, 95), 34)

    # Clouds
    def cloud(cx, cy, scale=1.0):
        parts = [(0, 0, 42), (35, -10, 34), (65, 0, 40), (30, 12, 38)]
        for ox, oy, rr in parts:
            pygame.draw.circle(screen, (245, 250, 255),
                               (int(cx + ox * scale), int(cy + oy * scale)), int(rr * scale))
        pygame.draw.ellipse(screen, (235, 242, 250), (cx - 10, cy + 10, int(110 * scale), int(35 * scale)))

    cloud(180, 120, 0.9)
    cloud(420, 80, 0.8)
    cloud(650, 140, 1.0)

    # Ground
    draw_vertical_gradient(screen, (0, GROUND_Y, WIDTH, HEIGHT - GROUND_Y),
                           (85, 195, 105), (35, 125, 70))

    # Ground texture lines
    for x in range(0, WIDTH, 55):
        pygame.draw.line(screen, (30, 105, 65), (x, GROUND_Y + 20), (x + 25, HEIGHT), 2)

class Basket:
    def __init__(self):
        self.w = 140
        self.h = 62
        self.x = WIDTH // 2
        self.y = GROUND_Y - 10
        self.speed = 1000

    @property
    def rect(self):
        return pygame.Rect(int(self.x - self.w // 2), int(self.y - self.h), self.w, self.h)

    def update(self, dt, keys):
        dx = 0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            dx -= 1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            dx += 1

        self.x += dx * self.speed * dt
        self.x = clamp(self.x, self.w // 2 + 10, WIDTH - self.w // 2 - 10)

    def draw(self):
        r = self.rect

        # Shadow
        sh = pygame.Surface((r.w + 40, 18), pygame.SRCALPHA)
        pygame.draw.ellipse(sh, (0, 0, 0, 80), sh.get_rect())
        screen.blit(sh, (r.x - 20, GROUND_Y - 8))

        # Basket body
        body = pygame.Surface((r.w, r.h), pygame.SRCALPHA)
        pygame.draw.rect(body, (185, 125, 70), (0, 6, r.w, r.h - 6), border_radius=16)
        pygame.draw.rect(body, (140, 90, 50), (0, 18, r.w, r.h - 18), border_radius=16)
        pygame.draw.rect(body, (230, 180, 120), (10, 6, r.w - 20, 12), border_radius=10)

        # Weave lines
        for i in range(10, r.w - 10, 14):
            pygame.draw.line(body, (115, 70, 38), (i, 22), (i, r.h - 8), 2)

        # Outline
        pygame.draw.rect(body, (60, 35, 20), (0, 6, r.w, r.h - 6), 3, border_radius=16)

        screen.blit(body, (r.x, r.y))

class Ball:
    def __init__(self, x, y, vx, vy, r):
        self.x = float(x)
        self.y = float(y)
        self.vx = float(vx)
        self.vy = float(vy)
        self.r = int(r)
        self.alive = True
        self.hit_ground = False

    def update(self, dt):
        self.vy += GRAVITY * dt
        self.x += self.vx * dt
        self.y += self.vy * dt

        if BOUNCE_WALLS:
            if self.x - self.r <= 0:
                self.x = self.r
                self.vx *= -BOUNCE_DAMP_X
            elif self.x + self.r >= WIDTH:
                self.x = WIDTH - self.r
                self.vx *= -BOUNCE_DAMP_X

            if self.y - self.r <= 0:
                self.y = self.r
                self.vy *= -BOUNCE_DAMP_Y

        if self.y + self.r >= GROUND_Y:
            self.hit_ground = True
            self.alive = False

    def draw(self):
        # Shadow
        dist = clamp((GROUND_Y - self.y) / 350.0, 0.0, 1.0)
        sw = int(self.r * (2.4 - 1.2 * dist))
        sh = int(self.r * (0.85 - 0.4 * dist))
        shadow = pygame.Surface((sw * 2, sh * 2), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0, 0, 0, 85), shadow.get_rect())
        screen.blit(shadow, (self.x - shadow.get_width() // 2, GROUND_Y - sh))

        # Ball
        pygame.draw.circle(screen, (230, 70, 70), (int(self.x), int(self.y)), self.r)
        pygame.draw.circle(screen, (255, 155, 155),
                           (int(self.x - self.r * 0.35), int(self.y - self.r * 0.35)),
                           int(self.r * 0.35))
        pygame.draw.circle(screen, (120, 25, 25), (int(self.x), int(self.y)), self.r, 3)

def draw_cannon(cx, cy, angle_rad):
    # Base
    pygame.draw.circle(screen, (45, 45, 55), (cx, cy), 26)
    pygame.draw.circle(screen, (25, 25, 30), (cx, cy), 26, 4)

    length = 95
    thickness = 18

    barrel = pygame.Surface((length + 40, thickness + 40), pygame.SRCALPHA)
    pygame.draw.rect(barrel, (95, 95, 110), (20, 20, length, thickness), border_radius=12)
    pygame.draw.rect(barrel, (40, 40, 50), (20, 20, length, thickness), 4, border_radius=12)

    # Pygame rotates counter-intuitive, so we rotate using degrees(-angle)
    rotated = pygame.transform.rotate(barrel, -math.degrees(angle_rad))
    rect = rotated.get_rect(center=(cx, cy))
    screen.blit(rotated, rect.topleft)

    # Muzzle point
    mx = cx + math.cos(angle_rad) * length
    my = cy + math.sin(angle_rad) * length

    # Muzzle glow
    glow = pygame.Surface((60, 60), pygame.SRCALPHA)
    pygame.draw.circle(glow, (255, 230, 180, 90), (30, 30), 18)
    screen.blit(glow, (mx - 30, my - 30))

    return mx, my

# GAME STATE
font = pygame.font.SysFont("consolas", 22)

basket = Basket()
balls = []
score = 0
misses = 0

# Cannon
CANNON_X = WIDTH // 2
CANNON_Y = 85

# Throw timers
spawn_timer = 0.0
next_spawn = BALL_THROW_DELAY

def compute_next_spawn():
    if RANDOMIZE_THROW:
        return random.uniform(BALL_THROW_DELAY * DELAY_RANDOM_MIN, BALL_THROW_DELAY * DELAY_RANDOM_MAX)
    return BALL_THROW_DELAY

running = True
last_shot_angle = math.radians(270)

while running:
    dt = clock.tick(120) / 1000.0
    keys = pygame.key.get_pressed()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_t:
                AI_MODE = not AI_MODE
                mode = "AI" if AI_MODE else "MANUAL"
                print(f"[MODE] {mode}")

    # Update basket
    if AI_MODE and AI_NET is not None and len(balls) > 0:
        target_ball = max(balls, key=lambda b: b.y)
        obs = make_obs(basket.x, target_ball)
        action = forward(AI_NET, obs)
        move_dir = -1 if action == 0 else (1 if action == 2 else 0)
        basket.x += move_dir * basket.speed * dt
        basket.x = clamp(basket.x, basket.w // 2 + 10, WIDTH - basket.w // 2 - 10)
    else:
        basket.update(dt, keys)

    spawn_timer += dt
    if spawn_timer >= next_spawn:
        spawn_timer = 0.0

        angle_deg = random.uniform(205, 335)
        angle = math.radians(angle_deg)
        last_shot_angle = angle

        speed = random.uniform(650, 1050)

        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed

        start_x = CANNON_X + math.cos(angle) * 80
        start_y = CANNON_Y + math.sin(angle) * 80

        balls.append(Ball(start_x, start_y, vx, vy, r=random.choice([12, 14, 16])))

        next_spawn = compute_next_spawn()

    brect = basket.rect
    for ball in balls:
        ball.update(dt)

        if ball.alive:
            ball_rect = pygame.Rect(int(ball.x - ball.r), int(ball.y - ball.r), ball.r * 2, ball.r * 2)
            if ball_rect.colliderect(brect) and ball.vy > 0:
                ball.alive = False
                ball.hit_ground = False
                score += 1

    alive = []
    for ball in balls:
        if not ball.alive:
            if ball.hit_ground:
                misses += 1
        else:
            alive.append(ball)
    balls = alive
    draw_background()
    t = pygame.time.get_ticks() / 1000.0
    wiggle = math.sin(t * 1.5) * math.radians(10)
    draw_cannon(CANNON_X, CANNON_Y, last_shot_angle + wiggle)

    for ball in balls:
        ball.draw()
    basket.draw()

    hud = font.render(f"Score: {score}   Misses: {misses}", True, WHITE)
    screen.blit(hud, (18, 16))

    rate = font.render(f"BALL_THROW_DELAY = {BALL_THROW_DELAY:.2f}s", True, (235, 240, 250))
    screen.blit(rate, (18, 44))

    tip = font.render(f"Mode: {'AI' if AI_MODE else 'MANUAL'}   (T to toggle)   Move: A/D", True, (235, 240, 250))
    screen.blit(tip, (18, 70))

    pygame.display.flip()

pygame.quit()
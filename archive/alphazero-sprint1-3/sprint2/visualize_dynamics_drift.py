import argparse
import time

import gymnasium as gym
import numpy as np
import pygame

from train_dynamics import load_dynamics_model


SCREEN_WIDTH = 700
SCREEN_HEIGHT = 420
CART_Y = 300
X_THRESHOLD = 2.4
THETA_THRESHOLD_RADIANS = 12 * 2 * np.pi / 360


def cartpole_done(state):
    x, _, theta, _ = state

    return (
        x < -X_THRESHOLD
        or x > X_THRESHOLD
        or theta < -THETA_THRESHOLD_RADIANS
        or theta > THETA_THRESHOLD_RADIANS
    )


def state_to_points(state):
    x, _, theta, _ = state

    world_width = X_THRESHOLD * 2
    scale = SCREEN_WIDTH / world_width

    cart_x = SCREEN_WIDTH / 2 + x * scale
    cart_y = CART_Y

    pole_length = 120
    pole_x = cart_x + pole_length * np.sin(theta)
    pole_y = cart_y - pole_length * np.cos(theta)

    return (cart_x, cart_y), (pole_x, pole_y)


def draw_cartpole(surface, state, color, label, y_offset=0):
    cart_center, pole_tip = state_to_points(state)
    cart_x, cart_y = cart_center
    pole_x, pole_y = pole_tip
    cart_y += y_offset
    pole_y += y_offset

    cart_x = int(round(cart_x))
    cart_y = int(round(cart_y))
    pole_x = int(round(pole_x))
    pole_y = int(round(pole_y))

    cart_rect = pygame.Rect(0, 0, 58, 28)
    cart_rect.center = (cart_x, cart_y)

    pygame.draw.rect(surface, color, cart_rect, border_radius=3)
    pygame.draw.line(surface, color, (cart_x, cart_y), (pole_x, pole_y), 7)
    pygame.draw.circle(surface, color, (cart_x, cart_y), 5)

    font = pygame.font.SysFont("Arial", 18)
    text = font.render(label, True, color)
    surface.blit(text, (cart_x - 28, cart_y + 24))


def draw_text(surface, lines):
    font = pygame.font.SysFont("Arial", 20)

    for index, line in enumerate(lines):
        text = font.render(line, True, (30, 30, 30))
        surface.blit(text, (20, 20 + index * 26))


def choose_action(env, step, mode):
    if mode == "alternate":
        return step % env.action_space.n

    return env.action_space.sample()


def visualize(args):
    model = load_dynamics_model(args.model_path)

    env = gym.make(args.env)
    env.action_space.seed(args.seed)

    true_state, _ = env.reset(seed=args.seed)
    pred_state = np.array(true_state, dtype=np.float32)

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("CartPole true state vs multi-step model prediction")
    clock = pygame.time.Clock()

    true_color = (20, 95, 190)
    pred_color = (215, 60, 45)

    running = True

    for step in range(args.steps + 1):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not running:
            break

        state_error = np.linalg.norm(np.array(true_state) - np.array(pred_state))

        screen.fill((246, 246, 242))
        pygame.draw.line(screen, (70, 70, 70), (0, CART_Y + 16), (SCREEN_WIDTH, CART_Y + 16), 2)

        draw_cartpole(screen, true_state, true_color, "true")
        draw_cartpole(screen, pred_state, pred_color, "pred")

        draw_text(
            screen,
            [
                f"step: {step}/{args.steps}",
                f"state error L2: {state_error:.4f}",
                "blue = real env state",
                "red = recursive model prediction"
            ]
        )

        pygame.display.flip()

        print(
            f"step={step:2d} | "
            f"error={state_error:.5f} | "
            f"true={np.array2string(np.array(true_state), precision=3)} | "
            f"pred={np.array2string(np.array(pred_state), precision=3)}"
        )

        if step == args.steps:
            time.sleep(args.pause_last)
            break

        time.sleep(args.pause)

        action = choose_action(env, step, args.action_mode)
        true_state, _, terminated, truncated, _ = env.step(action)
        pred_state, pred_reward = model.predict(pred_state, action)

        if args.stop_on_done and (terminated or truncated or cartpole_done(pred_state)):
            print(
                "stopped early | "
                f"true_done={terminated or truncated} | "
                f"pred_done={cartpole_done(pred_state)} | "
                f"pred_reward={pred_reward:.3f}"
            )
            break

        clock.tick(args.fps)

    env.close()
    pygame.quit()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize drift from recursive CartPole dynamics predictions."
    )

    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--model-path", default="dynamics_model_cartpole.pt")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-mode", choices=["random", "alternate"], default="random")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--pause", type=float, default=0.55)
    parser.add_argument("--pause-last", type=float, default=3.0)
    parser.add_argument("--stop-on-done", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    visualize(parse_args())

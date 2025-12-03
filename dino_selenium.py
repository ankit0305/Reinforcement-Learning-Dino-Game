import base64
import os
import time
from collections import deque
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import gymnasium as gym
import numpy as np
from PIL import Image
from gymnasium import spaces
from selenium import webdriver
from selenium.common.exceptions import WebDriverException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
import imageio
from tqdm import tqdm


class ChromeDinoEnv(gym.Env):
    """
    Improved Chrome T-Rex game environment for reinforcement learning.
    
    Features:
    - Better error handling and recovery
    - Optimized observation processing
    - Dynamic reward shaping
    - Improved game state detection
    - Automatic ChromeDriver management
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        screen_width: int = 96,
        screen_height: int = 96,
        chromedriver_path: Optional[str] = None,
        frame_stack: int = 4,
        headless: bool = False,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.chromedriver_path = chromedriver_path
        self.frame_stack = frame_stack
        self.headless = headless
        self.render_mode = render_mode
        
        # Metrics
        self.episode_score = 0
        self.last_score = 0
        self.num_steps = 0
        self.consecutive_failures = 0
        
        # Action space: 0=do nothing, 1=jump, 2=duck
        self.action_space = spaces.Discrete(3)
        
        # Observation space: grayscale image stack
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.screen_height, self.screen_width, self.frame_stack),
            dtype=np.uint8
        )
        
        # Initialize driver
        self._driver = None
        self._setup_driver()
        
        # Frame buffer
        self.state_queue = deque(maxlen=self.frame_stack)
        
        # Action mapping
        self.actions_map = {
            0: None,              # Do nothing
            1: Keys.SPACE,        # Jump
            2: Keys.ARROW_DOWN    # Duck
        }

    def _setup_driver(self):
        """Initialize Chrome WebDriver with automatic ChromeDriver management"""
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--mute-audio")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=800,600")
        chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
        
        if self.headless:
            chrome_options.add_argument("--headless=new")  # Use new headless mode

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use Selenium Manager for automatic driver management
                # This avoids ChromeDriver version mismatch issues
                if self.chromedriver_path:
                    # Only use custom path if explicitly provided and needed
                    from selenium.webdriver.chrome.service import Service
                    print(f"Using custom ChromeDriver at: {self.chromedriver_path}")
                    service = Service(executable_path=self.chromedriver_path)
                    self._driver = webdriver.Chrome(service=service, options=chrome_options)
                else:
                    # Let Selenium Manager handle ChromeDriver automatically
                    print("Using Selenium Manager for automatic ChromeDriver management...")
                    self._driver = webdriver.Chrome(options=chrome_options)
                
                print(f"✓ Chrome driver initialized successfully (attempt {attempt + 1})")
                return
                
            except Exception as e:
                error_msg = str(e)
                print(f"✗ Driver initialization attempt {attempt + 1} failed: {error_msg}")
                
                # Check for version mismatch
                if "version" in error_msg.lower() and "chromedriver" in error_msg.lower():
                    print("\n" + "="*70)
                    print("CHROMEDRIVER VERSION MISMATCH DETECTED")
                    print("="*70)
                    print("\nRecommended solutions:")
                    print("1. Remove custom ChromeDriver and let Selenium manage it:")
                    print("   rm /opt/homebrew/bin/chromedriver")
                    print("   # Then restart this script")
                    print("\n2. Or update Chrome to match your ChromeDriver:")
                    print("   # Update Chrome to version 143")
                    print("\n3. Or install webdriver-manager:")
                    print("   pip install webdriver-manager")
                    print("   # Then modify script to use ChromeDriverManager")
                    print("="*70 + "\n")
                
                if attempt < max_retries - 1:
                    print(f"Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    print("\n❌ Failed to initialize Chrome driver after multiple attempts")
                    print("Please ensure:")
                    print("  1. Google Chrome is installed")
                    print("  2. Chrome and ChromeDriver versions match")
                    print("  3. Or let Selenium Manager handle it automatically\n")
                    raise RuntimeError(
                        "Failed to initialize Chrome driver. "
                        "Try removing custom ChromeDriver from PATH and let Selenium manage it."
                    )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment and return initial observation (Gymnasium API)"""
        super().reset(seed=seed)
        
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Navigate to game
                self._driver.get('chrome://dino')
                
                # Wait for canvas
                WebDriverWait(self._driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "runner-canvas"))
                )
                
                # Reset metrics
                self.episode_score = 0
                self.last_score = 0
                self.num_steps = 0
                
                # Start game
                body = self._driver.find_element(By.TAG_NAME, "body")
                body.send_keys(Keys.SPACE)
                time.sleep(0.15)  # Wait for game to start
                
                # Initialize state queue
                self.state_queue.clear()
                obs = self._get_observation()
                
                # Fill frame stack
                for _ in range(self.frame_stack - 1):
                    self.state_queue.append(obs[:, :, 0])
                
                self.consecutive_failures = 0
                return obs, {}  # Gymnasium returns (obs, info)
                
            except Exception as e:
                print(f"Reset attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    self._recover_driver()
                else:
                    return self._get_blank_observation(), {}
        
        return self._get_blank_observation(), {}

    def _recover_driver(self):
        """Attempt to recover from driver errors"""
        try:
            if self._driver:
                self._driver.quit()
        except:
            pass
        
        self._driver = None
        self._setup_driver()

    def _capture_screenshot(self) -> np.ndarray:
        """Capture and process screenshot from game canvas"""
        try:
            canvas_data = self._driver.execute_script(
                "return document.querySelector('canvas.runner-canvas').toDataURL()"
            )
            
            if not canvas_data or not canvas_data.startswith("data:image/png;base64,"):
                raise ValueError("Invalid canvas data")
            
            # Decode image
            image_data = canvas_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = np.array(Image.open(BytesIO(image_bytes)))
            
            return image
            
        except Exception as e:
            print(f"Screenshot capture failed: {e}")
            self.consecutive_failures += 1
            return np.zeros((400, 600, 3), dtype=np.uint8)

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for observation"""
        # Convert to grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Crop to game area (remove score and other UI elements)
        image = image[120:400, 100:500]
        
        # Resize to target dimensions
        image = cv2.resize(image, (self.screen_width, self.screen_height), 
                          interpolation=cv2.INTER_AREA)
        
        # Apply slight contrast enhancement
        image = cv2.convertScaleAbs(image, alpha=1.2, beta=0)
        
        return image.astype(np.uint8)

    def _get_observation(self) -> np.ndarray:
        """Get current observation (stacked frames)"""
        raw_image = self._capture_screenshot()
        processed_image = self._preprocess_image(raw_image)
        
        self.state_queue.append(processed_image)
        
        # Stack frames
        if len(self.state_queue) < self.frame_stack:
            frames = [processed_image] * self.frame_stack
        else:
            frames = list(self.state_queue)
        
        return np.stack(frames, axis=-1)

    def _get_blank_observation(self) -> np.ndarray:
        """Return blank observation for error cases"""
        blank_frame = np.zeros((self.screen_height, self.screen_width), dtype=np.uint8)
        return np.stack([blank_frame] * self.frame_stack, axis=-1)

    def _get_score(self) -> int:
        """Get current game score"""
        try:
            score_digits = self._driver.execute_script(
                "return Runner.instance_.distanceMeter.digits"
            )
            if score_digits:
                return int(''.join(map(str, score_digits)))
        except:
            pass
        return self.last_score

    def _is_game_over(self) -> bool:
        """Check if game is over"""
        try:
            return bool(self._driver.execute_script("return Runner.instance_.crashed"))
        except:
            return self.consecutive_failures > 5

    def _calculate_reward(self, score: int, done: bool) -> float:
        """
        Calculate reward with shaping for better learning.
        
        Reward components:
        - Progress reward: small positive for advancing
        - Survival reward: continuous small reward
        - Death penalty: negative reward for dying
        - Score improvement: reward for increasing score
        """
        if done:
            reward = -10.0
        else:
            reward = 0.1
            
            # Progress reward (score increase)
            score_delta = score - self.last_score
            if score_delta > 0:
                reward += score_delta * 0.01
        
        self.last_score = score
        return reward

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action and return transition (Gymnasium API with terminated/truncated)"""
        try:
            # Execute action
            if action in self.actions_map and self.actions_map[action] is not None:
                body = self._driver.find_element(By.TAG_NAME, "body")
                body.send_keys(self.actions_map[action])
            
            # Small delay for action to take effect
            time.sleep(0.04)
            
            # Get next state
            obs = self._get_observation()
            terminated = self._is_game_over()
            truncated = False  # Game doesn't have time limits
            score = self._get_score()
            
            # Calculate reward
            reward = self._calculate_reward(score, terminated)
            
            self.num_steps += 1
            self.episode_score = score
            
            info = {
                "score": score,
                "steps": self.num_steps,
                "episode_score": self.episode_score
            }
            
            return obs, reward, terminated, truncated, info
            
        except Exception as e:
            print(f"Step execution failed: {e}")
            obs = self._get_blank_observation()
            return obs, -1.0, True, False, {"score": 0, "error": str(e)}

    def render(self):
        """Render environment (Gymnasium API)"""
        if self.render_mode is None:
            return None
            
        try:
            img = self._capture_screenshot()
            
            if self.render_mode == 'rgb_array':
                if len(img.shape) == 3:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img
            elif self.render_mode == 'human':
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.imshow('Chrome Dino', img)
                cv2.waitKey(1)
                return None
                
        except Exception as e:
            print(f"Render failed: {e}")
            return None

    def close(self):
        """Clean up resources"""
        if self._driver:
            try:
                self._driver.quit()
            except:
                pass
            self._driver = None
        
        # Close OpenCV windows if any
        cv2.destroyAllWindows()


def make_env(rank: int, config: Dict[str, Any]):
    """Create and wrap environment"""
    def _init():
        env = ChromeDinoEnv(**config)
        env = Monitor(env)
        return env
    return _init


def train_model(
    total_timesteps: int = 500000,
    num_envs: int = 1,
    save_dir: str = "./models",
    checkpoint_freq: int = 50000,
    eval_freq: int = 10000,
    chromedriver_path: Optional[str] = None
):
    """Train PPO model with improved configuration"""
    
    # Create directories
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_path / "checkpoints"
    checkpoint_path.mkdir(exist_ok=True)
    
    # Environment configuration
    env_config = {
        "screen_width": 96,
        "screen_height": 96,
        "chromedriver_path": chromedriver_path,
        "frame_stack": 4,
        "headless": False,
        "render_mode": None
    }
    
    print(f"Creating {num_envs} training environments...")
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_envs)])
    env = VecMonitor(env)
    
    # Create evaluation environment
    eval_config = env_config.copy()
    eval_config["render_mode"] = None
    eval_env = ChromeDinoEnv(**eval_config)
    eval_env = Monitor(eval_env)
    
    print("Initializing PPO model...")
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=str(save_path / "tensorboard")
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq // num_envs,
        save_path=str(checkpoint_path),
        name_prefix='dino_model'
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path / "best_model"),
        log_path=str(save_path / "eval_logs"),
        eval_freq=eval_freq // num_envs,
        deterministic=True,
        render=False
    )
    
    print("Starting training...")
    print("="*70)
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback]
    )
    
    # Save final model
    final_path = save_path / "final_model"
    model.save(str(final_path))
    print("="*70)
    print(f"✓ Training complete! Model saved to {final_path}")
    
    env.close()
    eval_env.close()
    
    return model


def evaluate_model(
    model_path: str,
    num_episodes: int = 10,
    save_gif: bool = True,
    chromedriver_path: Optional[str] = None
):
    """Evaluate trained model and optionally save demonstration"""
    
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    env = ChromeDinoEnv(
        screen_width=96,
        screen_height=96,
        chromedriver_path=chromedriver_path,
        headless=False,
        render_mode='rgb_array' if save_gif else None
    )
    
    scores = []
    images = [] if save_gif else None
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        while not (terminated or truncated):
            # Predict action
            action, _ = model.predict(obs, deterministic=True)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # Capture frame for GIF
            if save_gif and episode == 0:
                frame = env.render()
                if frame is not None:
                    images.append(frame)
        
        score = info.get('score', 0)
        scores.append(score)
        print(f"Episode {episode + 1} - Score: {score}, Reward: {episode_reward:.2f}")
    
    env.close()
    
    # Print statistics
    print(f"\n{'='*50}")
    print(f"Evaluation Results ({num_episodes} episodes):")
    print(f"Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Max Score: {np.max(scores)}")
    print(f"Min Score: {np.min(scores)}")
    print(f"{'='*50}")
    
    # Save GIF
    if save_gif and images:
        gif_path = "dino_demo.gif"
        print(f"\nSaving demonstration to {gif_path}...")
        imageio.mimsave(gif_path, images, fps=20)
        print(f"✓ GIF saved successfully!")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Chrome Dino RL Training')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'eval', 'both'],
                       help='Mode: train, eval, or both')
    parser.add_argument('--timesteps', type=int, default=500000,
                       help='Total training timesteps')
    parser.add_argument('--num-envs', type=int, default=1,
                       help='Number of parallel environments')
    parser.add_argument('--model-path', type=str, default='./models/final_model',
                       help='Path to model for evaluation')
    parser.add_argument('--chromedriver', type=str, default=None,
                       help='Path to chromedriver executable (optional, Selenium Manager handles it)')
    parser.add_argument('--eval-episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    
    args = parser.parse_args()
    
    if args.mode in ['train', 'both']:
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70 + "\n")
        model = train_model(
            total_timesteps=args.timesteps,
            num_envs=args.num_envs,
            chromedriver_path=args.chromedriver
        )
    
    if args.mode in ['eval', 'both']:
        print("\n" + "="*70)
        print("STARTING EVALUATION")
        print("="*70 + "\n")
        evaluate_model(
            model_path=args.model_path,
            num_episodes=args.eval_episodes,
            save_gif=True,
            chromedriver_path=args.chromedriver
        )


if __name__ == '__main__':
    main()

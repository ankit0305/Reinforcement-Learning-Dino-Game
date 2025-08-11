import base64
import os
import time
from collections import deque
from io import BytesIO

import cv2
import gym
import numpy as np
from PIL import Image
from gym import spaces
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
import imageio
from tqdm import tqdm


class EnvironmentChromeTRex(gym.Env):
    """Chrome T-Rex game environment for reinforcement learning"""

    def __init__(self,
                 screen_width: int,  # width of the compressed image
                 screen_height: int,  # height of the compressed image
                 chromedriver_path: str = 'chromedriver'
                 ):
        super(EnvironmentChromeTRex, self).__init__()
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.chromedriver_path = chromedriver_path
        self.num_observation = 0
        self.viewer = None

        # Action space: 0=do nothing, 1=jump, 2=duck
        self.action_space = spaces.Discrete(3)
        
        # Observation space: grayscale images stacked (4 frames)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.screen_height, self.screen_width, 4),  # Fixed order: height, width, channels
            dtype=np.uint8
        )
        
        # Initialize Chrome driver
        self._setup_driver()
        
        # Current state represented by 4 images
        self.state_queue = deque(maxlen=4)

        # Action mapping
        self.actions_map = [
            None,              # 0: do nothing
            Keys.SPACE,        # 1: jump (SPACE is more reliable than ARROW_UP)
            Keys.ARROW_DOWN    # 2: duck
        ]

    def _setup_driver(self):
        """Initialize Chrome WebDriver with appropriate options"""
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--mute-audio")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        # chrome_options.add_argument("--headless")  # Uncomment for headless mode

        try:
            from selenium.webdriver.chrome.service import Service
            
            # Use Service class for newer versions of Selenium
            if self.chromedriver_path and self.chromedriver_path != 'chromedriver':
                service = Service(executable_path=self.chromedriver_path)
                self._driver = webdriver.Chrome(service=service, options=chrome_options)
            else:
                # Let selenium find chromedriver automatically (must be in PATH)
                self._driver = webdriver.Chrome(options=chrome_options)
                
        except ImportError:
            # Fallback for older selenium versions
            try:
                self._driver = webdriver.Chrome(
                    executable_path=self.chromedriver_path,
                    options=chrome_options
                )
            except Exception as e:
                print(f"Failed to initialize Chrome driver with executable_path: {e}")
                # Try without executable_path (chromedriver must be in PATH)
                self._driver = webdriver.Chrome(options=chrome_options)
        except Exception as e:
            print(f"Failed to initialize Chrome driver: {e}")
            raise

    def reset(self):
        """Reset the environment and return initial observation"""
        try:
            self._driver.get('chrome://dino')
        except WebDriverException as e:
            print(f"Error loading dino page: {e}")
            return self._get_blank_observation()

        try:
            # Wait for canvas to load
            WebDriverWait(self._driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "runner-canvas"))
            )

            # Start the game
            body = self._driver.find_element(By.TAG_NAME, "body")
            body.send_keys(Keys.SPACE)
            
            # Clear state queue and get initial observation
            self.state_queue.clear()
            time.sleep(0.1)  # Give game time to start
            
            return self._next_observation()
            
        except Exception as e:
            print(f"Error during reset: {e}")
            return self._get_blank_observation()

    def _get_image(self):
        """Capture screenshot from the game canvas"""
        try:
            LEADING_TEXT = "data:image/png;base64,"
            canvas_data = self._driver.execute_script(
                "return document.querySelector('canvas.runner-canvas').toDataURL()"
            )
            
            if not canvas_data or not canvas_data.startswith(LEADING_TEXT):
                raise ValueError("Invalid canvas data")
                
            image_data = canvas_data[len(LEADING_TEXT):]
            image_bytes = base64.b64decode(image_data)
            image = np.array(Image.open(BytesIO(image_bytes)))
            
            return image
            
        except Exception as e:
            print(f"Error capturing image: {e}")
            # Return a blank image if capture fails
            return np.zeros((400, 600, 3), dtype=np.uint8)

    def _next_observation(self):
        """Process the current image and return stacked observation"""
        image = self._get_image()
        
        # Convert to grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Crop to relevant game area (adjust these values based on your screen)
        image = image[120:400, 100:500]  # Crop to game area
        
        # Resize to desired dimensions
        image = cv2.resize(image, (self.screen_width, self.screen_height))
        
        # Normalize pixel values
        image = image.astype(np.uint8)
        
        self.num_observation += 1
        self.state_queue.append(image)

        # Stack 4 frames
        if len(self.state_queue) < 4:
            # Fill with copies of current frame if we don't have 4 yet
            stacked_frames = [image] * 4
        else:
            stacked_frames = list(self.state_queue)
        
        return np.stack(stacked_frames, axis=-1)

    def _get_blank_observation(self):
        """Return a blank observation when errors occur"""
        blank_frame = np.zeros((self.screen_height, self.screen_width), dtype=np.uint8)
        return np.stack([blank_frame] * 4, axis=-1)

    def _get_score(self):
        """Get current game score"""
        try:
            score_digits = self._driver.execute_script(
                "return Runner.instance_.distanceMeter.digits"
            )
            if score_digits:
                return int(''.join(map(str, score_digits)))
        except:
            pass
        return 0

    def _get_done(self):
        """Check if game is over"""
        try:
            return self._driver.execute_script("return Runner.instance_.crashed")
        except:
            return False

    def step(self, action: int):
        """Execute action and return next observation, reward, done, info"""
        try:
            # Execute action
            if action > 0 and action < len(self.actions_map):
                key = self.actions_map[action]
                if key is not None:
                    body = self._driver.find_element(By.TAG_NAME, "body")
                    body.send_keys(key)

            # Small delay to let action take effect
            time.sleep(0.05)

            # Get next observation
            obs = self._next_observation()
            
            # Check if game is done
            done = self._get_done()
            
            # Calculate reward
            if done:
                reward = -10  # Penalty for dying
            else:
                reward = 0.1  # Small positive reward for surviving
            
            # Get current score for info
            score = self._get_score()

            return obs, reward, done, {"score": score}
            
        except Exception as e:
            print(f"Error during step: {e}")
            obs = self._get_blank_observation()
            return obs, -1, True, {"score": 0}

    def render(self, mode: str = 'human'):
        """Render the environment"""
        try:
            img = self._get_image()
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if mode == 'rgb_array':
                return img
            elif mode == 'human':
                try:
                    from gym.envs.classic_control import rendering
                    if self.viewer is None:
                        self.viewer = rendering.SimpleImageViewer()
                    self.viewer.imshow(img)
                    return self.viewer.isopen
                except ImportError:
                    print("Rendering requires gym[classic_control]")
                    return True
        except Exception as e:
            print(f"Error during rendering: {e}")
            return None

    def close(self):
        """Clean up resources"""
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        
        if hasattr(self, '_driver') and self._driver:
            try:
                self._driver.quit()
            except:
                pass


def main():
    """Main training and testing function"""
    
    # Configuration
    SCREEN_WIDTH = 96
    SCREEN_HEIGHT = 96
    CHROMEDRIVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "/Users/ankitjaiswal/Downloads/chromedriver-mac-arm64/chromedriver")
    
    # Environment factory
    def make_env():
        return EnvironmentChromeTRex(
            screen_width=SCREEN_WIDTH,
            screen_height=SCREEN_HEIGHT,
            chromedriver_path=CHROMEDRIVER_PATH
        )

    # Training configuration
    DO_TRAIN = True
    NUM_CPU = 1  # Start with 1 CPU to debug
    TOTAL_TIMESTEPS = 100000  # Reduced for testing
    SAVE_PATH = "chrome_dino_ppo_model"
    
    if DO_TRAIN:
        print("Creating training environment...")
        env = SubprocVecEnv([make_env for _ in range(NUM_CPU)])
        
        print("Initializing PPO model...")
        model = PPO(
            "CnnPolicy",  # Use string instead of imported class
            env,
            verbose=1,
            # tensorboard_log="./tensorboard_logs/",  # Commented out to avoid tensorboard dependency
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
        )
        
        print("Starting training...")
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path='./checkpoints/',
            name_prefix='chrome_dino_checkpoint'
        )
        
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback]
        )
        
        print(f"Saving model to {SAVE_PATH}")
        model.save(SAVE_PATH)
        env.close()
    
    # Testing/Demo
    print("Loading model for demonstration...")
    try:
        env = make_env()
        model = PPO.load(SAVE_PATH)
        
        print("Running demonstration...")
        images = []
        obs = env.reset()
        
        for i in tqdm(range(100)):  # Reduced for testing
            # Render and save frame
            img = env.render(mode='rgb_array')
            if img is not None:
                images.append(img)
            
            # Predict action
            action, _ = model.predict(obs, deterministic=True)
            
            # Take step
            obs, reward, done, info = env.step(action)
            
            if done:
                print(f"Game over! Final score: {info.get('score', 0)}")
                break
        
        env.close()
        
        # Save GIF if we have frames
        if images:
            print("Saving demonstration GIF...")
            imageio.mimsave('dino_demo.gif', images, fps=10)
            print("GIF saved as 'dino_demo.gif'")
            
    except Exception as e:
        print(f"Error during demonstration: {e}")


if __name__ == '__main__':
    main()
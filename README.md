# Chrome Dino Game Reinforcement Learning with PPO

This project trains a reinforcement learning (RL) agent to play the **Chrome Dino** (T-Rex runner) game using **Stable-Baselines3 PPO** and a custom **OpenAI Gym** environment implemented with Selenium.

The agent learns by interacting with the Chrome Dino game in real-time, receiving rewards for survival and penalties for collisions.

---

## 📌 Features

* Custom **Gym environment** for Chrome Dino using Selenium
* Supports **jump** and **duck** actions
* Captures game frames directly from the Chrome canvas
* Preprocesses and stacks **4 grayscale frames** for temporal context
* Train with **Stable-Baselines3 PPO**
* Option to save **GIF** of the trained model’s performance
* Includes **checkpoint saving** during training

---

## ⚙️ Requirements

Install the dependencies with:

```bash
pip install selenium stable-baselines3 opencv-python-headless gym pillow tqdm imageio
```

You also need:

* **Google Chrome** installed
* **Matching ChromeDriver** for your Chrome version
  Download: [https://chromedriver.chromium.org/downloads](https://chromedriver.chromium.org/downloads)
  Make sure the `chromedriver` binary is either:

  * In your system `PATH`, or
  * Provided via the `chromedriver_path` parameter in the script

---

## 📂 Project Structure

```
.
├── chrome_dino_rl.py      # Main script (environment + training + demo)
├── checkpoints/           # Saved PPO checkpoints
├── chrome_dino_ppo_model  # Final trained model
└── dino_demo.gif          # Demo of the agent (generated after training)
```

---

## 🔄 RL Workflow Diagram

```mermaid
flowchart TD
    A[Start Script] --> B[Initialize Environment]
    B --> C[Launch Chrome & Open Dino Game]
    C --> D[Capture Game Frame from Canvas]
    D --> E[Preprocess: Crop → Grayscale → Resize]
    E --> F[Stack 4 Frames as Observation]
    F --> G[PPO Model Chooses Action]
    G --> H[Send Action via Selenium (Jump/Duck/None)]
    H --> I[Environment Updates Game State]
    I --> J[Calculate Reward (+0.1 survival, -10 crash)]
    J --> K[Check if Game Over]
    K -->|No| D
    K -->|Yes| L[Reset Environment]
    L --> M[Continue Training or Demo Loop]
```

---

## 🚀 How It Works

1. **Environment**

   * Implemented in `EnvironmentChromeTRex` (inherits from `gym.Env`)
   * Uses Selenium to launch Chrome and navigate to `chrome://dino`
   * Captures frames from the game’s HTML canvas
   * Converts frames to grayscale, crops, resizes, and stacks them for observation
   * Action space:

     ```
     0 → do nothing
     1 → jump
     2 → duck
     ```

2. **Training**

   * Uses PPO from Stable-Baselines3
   * Rewards:

     * `+0.1` per time step alive
     * `-10` when game over
   * Checkpoints saved every 10,000 steps

3. **Testing/Demo**

   * Loads the trained PPO model
   * Runs the game with model predictions
   * Optionally records frames and saves a GIF

---

## ▶️ Running the Code

### 1. Training the Agent

```bash
python chrome_dino_rl.py
```

* Set `DO_TRAIN = True` inside `main()` to enable training
* Adjust `TOTAL_TIMESTEPS` for longer training sessions

### 2. Testing the Agent

```bash
python chrome_dino_rl.py
```

* Set `DO_TRAIN = False` to skip training and only run a demo

---

## 📊 Example PPO Hyperparameters

```python
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
)
```

---

## 🎥 Demo Output

After testing, the script can generate a `dino_demo.gif` showing the trained agent playing:

```
Saving demonstration GIF...
GIF saved as 'dino_demo.gif'
```

---

## 🛠 Troubleshooting

* **ChromeDriver version mismatch** → Update ChromeDriver to match your Chrome browser version.
* **Headless mode issues** → Disable `chrome_options.add_argument("--headless")` if frames are not captured correctly.
* **Selenium errors** → Ensure Chrome and ChromeDriver are correctly installed and accessible.

---

## 📜 License

MIT License — feel free to modify and share.

---

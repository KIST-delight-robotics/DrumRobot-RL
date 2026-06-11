# drum_robot/tasks/legacy_task/agents/skrl_ppo_pr_cfg.py

def get_default_ppo_cfg():
    return {
        "seed": 42,
        "models": {
            "policy": {
                "class": "GaussianMixin",
                "network": [
                    {
                        "name": "net",
                        "input": "OBSERVATIONS",
                        "layers": [256, 160, 128],
                        
                        "activations": ["elu", "elu", "elu"],
                    }
                ],
                "output": "ACTIONS",   # action_dim=9
            },
            "value": {
                "class": "DeterministicMixin",
                "network": [
                    {
                        "name": "net",
                        "input": "OBSERVATIONS",
                        "layers": [256, 160, 128],
                        "activations": ["elu", "elu", "elu"],
                    }
                ],
                "output": "ONE",       # value_dim=1
            },
        },
        "agent": {
            "class": "PPO",
            "experiment": {"directory": "drum_robot", "experiment_name": "point_reaching"},
            "rollouts": 16,
            "learning_epochs": 4,
            "mini_batches": 4,
            "discount_factor": 0.99,
            "lambda": 0.95,
            "learning_rate": 1e-4,
            "grad_norm_clip": 1.0,
            "ratio_clip": 0.2,
            "value_clip": 0.2,
            "clip_predicted_values": True,
            "entropy_loss_scale": 0.002,
            "value_loss_scale": 1.0,
            "kl_threshold": 0.008,
        },
        "trainer": {
            "class": "SequentialTrainer",
            "timesteps": 800000,
            "checkpoint_interval": "auto",
            "close_environment_at_exit": False,
        },
    }

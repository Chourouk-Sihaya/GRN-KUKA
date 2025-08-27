
"""
Gene Regulatory Network (GRN) Approach for Robotic Grasping

A bio-inspired control approach using gene regulatory networks (GRN)
for KUKA robotic arm grasping tasks, specifically KukaDiverseObjectEnv.
This approach implements Evolutionary Algorithm to evolve GRN architectures.

Author: [Chourouk Guettas]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from dataclasses import dataclass
from typing import Dict, List, Optional
import random
import time
import os
import matplotlib.pyplot as plt
from datetime import datetime
from copy import deepcopy
import pickle
import json

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resize = T.Compose([
    T.ToPILImage(),
    T.Resize(40, interpolation=Image.BICUBIC),
    T.ToTensor()
])

#Extract and preprocess visual state from environment
def get_screen(env):

    screen = env._get_observation().transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    return resize(screen).unsqueeze(0).to(device)

#Gene Regulatory Network parameters Genome
class GRNGenome:
    regulatory_matrix: np.ndarray = None
    sensor_input_weights: np.ndarray = None
    motor_output_weights: np.ndarray = None
    activation_thresholds: np.ndarray = None
    decay_rates: np.ndarray = None
    interaction_strengths: np.ndarray = None

    num_genes: int = 50
    num_actuators: int = 7
    sensor_dims: int = 8

    fitness: float = 0.0
    success_rate: float = 0.0
    avg_reward: float = 0.0
    generation: int = 0

    def __post_init__(self):
        if self.regulatory_matrix is None:
            self.initialize_random_grn()

    def initialize_random_grn(self):
        self.regulatory_matrix = np.random.randn(self.num_genes, self.num_genes) * 0.1
        sparsity_mask = np.random.random((self.num_genes, self.num_genes)) < 0.2
        self.regulatory_matrix *= sparsity_mask

        self.sensor_input_weights = np.random.randn(self.sensor_dims, self.num_genes) * 0.3
        self.motor_output_weights = np.random.randn(self.num_genes, self.num_actuators) * 0.2

        self.activation_thresholds = np.random.uniform(0.2, 0.8, self.num_genes)
        self.decay_rates = np.random.uniform(0.1, 0.9, self.num_genes)
        self.interaction_strengths = np.random.uniform(0.5, 2.0, self.num_genes)


class GeneRegulatoryController:
    def __init__(self, genome: GRNGenome):
        self.genome = genome
        self.gene_expression = np.ones(genome.num_genes) * 0.3
        self.expression_history = []
        self.dt = 0.05
        self.noise_level = 0.01

    def reset_episode(self):
        self.gene_expression = np.ones(self.genome.num_genes) * 0.3
        self.expression_history.clear()

    def hill_function(self, x, threshold, n=2.0):
        return (x**n) / (threshold**n + x**n)

    def process_sensor_input(self, visual_features, proprioception):
        sensor_vector = np.concatenate([
            visual_features.flatten()[:4],
            proprioception[:4]
        ])

        if len(sensor_vector) != self.genome.sensor_dims:
            sensor_vector = np.pad(sensor_vector, (0, max(0, self.genome.sensor_dims - len(sensor_vector))))
            sensor_vector = sensor_vector[:self.genome.sensor_dims]

        return sensor_vector

    def update_gene_expression(self, sensor_inputs):
        env_influence = np.tanh(np.dot(sensor_inputs, self.genome.sensor_input_weights))
        regulatory_input = np.dot(self.genome.regulatory_matrix, self.gene_expression)
        regulatory_input *= self.genome.interaction_strengths

        new_expression = np.zeros_like(self.gene_expression)

        for i in range(self.genome.num_genes):
            total_input = regulatory_input[i] + env_influence[i]

            activation = self.hill_function(
                np.abs(total_input),
                self.genome.activation_thresholds[i],
                n=2.0
            )

            if total_input < 0:
                activation = -activation

            expression_derivative = (
                activation -
                self.genome.decay_rates[i] * self.gene_expression[i] +
                np.random.normal(0, self.noise_level)
            )

            new_expression[i] = self.gene_expression[i] + self.dt * expression_derivative

        self.gene_expression = np.clip(new_expression, 0.0, 1.0)
        self.expression_history.append(self.gene_expression.copy())

    def generate_motor_commands(self):
        raw_commands = np.dot(self.gene_expression, self.genome.motor_output_weights)
        return np.tanh(raw_commands)

    def get_action(self, visual_state, proprioception=None):
        if proprioception is None:
            proprioception = np.zeros(4)

        sensor_inputs = self.process_sensor_input(visual_state, proprioception)
        self.update_gene_expression(sensor_inputs)
        action = self.generate_motor_commands()

        return action

#Lightweight visual feature extraction for GRN input
class GRNVisualEncoder:

    def __init__(self):
        self.conv1 = nn.Conv2d(3, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))

        self.conv1 = self.conv1.to(device)
        self.conv2 = self.conv2.to(device)
        self.pool = self.pool.to(device)

    def extract_features(self, visual_input):
        with torch.no_grad():
            x = F.relu(self.conv1(visual_input))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            features = x.flatten().cpu().numpy()

        return features / (np.linalg.norm(features) + 1e-8)

class GRNEvolutionaryOptimizer:

    def __init__(self, population_size=16, elite_fraction=0.25, mutation_rate=0.2):
        self.population_size = population_size
        self.elite_count = int(elite_fraction * population_size)
        self.mutation_rate = mutation_rate
        self.population = []
        self.generation = 0

        self.fitness_history = []
        self.success_history = []
        self.best_genome_history = []

        self.structural_mutation_rate = 0.1
        self.parameter_mutation_std = 0.1

    def initialize_population(self, env):
        self.visual_encoder = GRNVisualEncoder()

        env.reset()
        dummy_visual = get_screen(env)
        test_features = self.visual_encoder.extract_features(dummy_visual)

        self.population = []

        for i in range(self.population_size):
            genome = GRNGenome()
            genome.generation = 0

            np.random.seed(42 + i)
            genome.initialize_random_grn()
            
            self.population.append(genome)

    def evaluate_genome(self, genome: GRNGenome, env, episodes=5):
        controller = GeneRegulatoryController(genome)

        episode_rewards = []
        success_count = 0

        for episode in range(episodes):
            env.reset()
            controller.reset_episode()

            visual_state = get_screen(env)
            visual_features = self.visual_encoder.extract_features(visual_state)

            episode_reward = 0
            done = False
            steps = 0
            max_steps = 30

            while not done and steps < max_steps:
                steps += 1

                try:
                    action = controller.get_action(visual_features)
                    _, reward, done, _ = env.step(action)
                    episode_reward += reward

                    visual_state = get_screen(env)
                    visual_features = self.visual_encoder.extract_features(visual_state)

                    if reward > 0:
                        success_count += 1
                        break

                except Exception:
                    break

            episode_rewards.append(episode_reward)

        if episode_rewards:
            avg_reward = np.mean(episode_rewards)
            success_rate = success_count / episodes
            reward_std = np.std(episode_rewards)
            consistency = 1.0 / (1.0 + reward_std) if reward_std > 0 else 1.0

            fitness = (
                avg_reward +
                10.0 * success_rate +
                2.0 * consistency +
                1.0 * min(success_rate, 0.8)
            )
        else:
            fitness = -10.0
            success_rate = 0.0
            avg_reward = 0.0

        genome.fitness = fitness
        genome.success_rate = success_rate
        genome.avg_reward = avg_reward

        return fitness

    def mutate_grn(self, genome: GRNGenome):
        mutated = deepcopy(genome)

        try:
            # Structural mutations
            if random.random() < self.structural_mutation_rate:
                i, j = random.randint(0, genome.num_genes-1), random.randint(0, genome.num_genes-1)
                if abs(mutated.regulatory_matrix[i, j]) < 0.01:
                    mutated.regulatory_matrix[i, j] = random.gauss(0, 0.1)
                else:
                    mutated.regulatory_matrix[i, j] = 0

            # Parameter mutations
            for param_name in ['regulatory_matrix', 'sensor_input_weights', 'motor_output_weights']:
                if random.random() < self.mutation_rate:
                    param = getattr(mutated, param_name)
                    mutation = np.random.normal(0, self.parameter_mutation_std, param.shape)
                    setattr(mutated, param_name, param + mutation)

            # Gene-specific parameter mutations
            for param_name in ['activation_thresholds', 'decay_rates', 'interaction_strengths']:
                if random.random() < self.mutation_rate:
                    param = getattr(mutated, param_name)
                    mutation = np.random.normal(0, self.parameter_mutation_std, param.shape)
                    new_param = param + mutation

                    if param_name == 'activation_thresholds':
                        new_param = np.clip(new_param, 0.1, 1.0)
                    elif param_name == 'decay_rates':
                        new_param = np.clip(new_param, 0.05, 1.0)
                    elif param_name == 'interaction_strengths':
                        new_param = np.clip(new_param, 0.1, 3.0)

                    setattr(mutated, param_name, new_param)

            return mutated

        except Exception:
            return genome

    def evolve(self, env, max_generations=20):
        self.initialize_population(env)

        for generation in range(max_generations):
            self.generation = generation

            # Evaluate all genomes
            for genome in self.population:
                self.evaluate_genome(genome, env, episodes=5)

            # Sort by fitness
            self.population.sort(key=lambda x: x.fitness, reverse=True)

            # Track progress
            best = self.population[0]
            self.fitness_history.append(best.fitness)
            self.success_history.append(best.success_rate)
            self.best_genome_history.append(deepcopy(best))

            # Early stopping check
            if best.success_rate >= 0.7:
                robust_fitness = self.evaluate_genome(best, env, episodes=20)
                if best.success_rate >= 0.5:
                    break

            # Create next generation
            if generation < max_generations - 1:
                next_gen = []
                elites = deepcopy(self.population[:self.elite_count])
                next_gen.extend(elites)

                while len(next_gen) < self.population_size:
                    parent = random.choice(elites)
                    offspring = self.mutate_grn(parent)
                    offspring.generation = self.generation + 1
                    next_gen.append(offspring)

                self.population = next_gen

        return self.population[0]

def test_grn_genome(env, genome: GRNGenome, episodes=40):
    controller = GeneRegulatoryController(genome)
    visual_encoder = GRNVisualEncoder()

    rewards = []
    successes = 0
    episode_lengths = []

    for episode in range(episodes):
        controller.reset_episode()

        try:
            env.reset()
            visual_state = get_screen(env)
            visual_features = visual_encoder.extract_features(visual_state)

            episode_reward = 0
            done = False
            steps = 0
            max_steps = 30

            while not done and steps < max_steps:
                steps += 1

                action = controller.get_action(visual_features)
                _, reward, done, _ = env.step(action)
                episode_reward += reward

                visual_state = get_screen(env)
                visual_features = visual_encoder.extract_features(visual_state)

                if reward > 0:
                    successes += 1
                    break

            rewards.append(episode_reward)
            episode_lengths.append(steps)

        except Exception:
            rewards.append(0.0)
            episode_lengths.append(0)

    results = {
        'success_rate': successes / episodes if episodes > 0 else 0,
        'avg_reward': np.mean(rewards) if rewards else 0,
        'std_reward': np.std(rewards) if rewards else 0,
        'avg_length': np.mean(episode_lengths) if episode_lengths else 0,
        'all_rewards': rewards,
        'episode_lengths': episode_lengths,
        'total_successes': successes
    }

    return results

#Create video with GRN visualization
def create_grn_video(env, genome: GRNGenome, video_path, episodes=10):
    try:
        from PIL import Image, ImageDraw, ImageFont
        frames = []

        controller = GeneRegulatoryController(genome)
        visual_encoder = GRNVisualEncoder()

        for episode in range(episodes):
            controller.reset_episode()
            env.reset()

            visual_state = get_screen(env)
            visual_features = visual_encoder.extract_features(visual_state)

            episode_reward = 0
            done = False
            steps = 0
            max_steps = 80

            while not done and steps < max_steps:
                steps += 1

                action = controller.get_action(visual_features)

                if steps % 2 == 0:
                    try:
                        img = env.render(mode='rgb_array')
                        frame_with_overlay = add_grn_overlay(
                            img, controller.gene_expression, episode+1, episodes,
                            episode_reward, steps, genome
                        )
                        frames.append(frame_with_overlay)
                    except Exception:
                        pass

                _, reward, done, _ = env.step(action)
                episode_reward += reward

                if reward > 0:
                    for _ in range(8):
                        try:
                            img = env.render(mode='rgb_array')
                            success_overlay = add_success_overlay(img, episode+1, steps, reward)
                            frames.append(success_overlay)
                        except:
                            pass
                    break

                visual_state = get_screen(env)
                visual_features = visual_encoder.extract_features(visual_state)

        # Create video
        if frames:
            try:
                from moviepy.editor import ImageSequenceClip
                clip = ImageSequenceClip(frames, fps=10)
                clip.write_videofile(video_path, codec='libx264', audio=False, verbose=False, logger=None)
                return len(frames)
            except ImportError:
                gif_path = video_path.replace('.mp4', '.gif')
                images = [Image.fromarray(frame) for frame in frames[::2]]
                images[0].save(gif_path, save_all=True, append_images=images[1:], duration=200, loop=0)
                return len(frames)

        return 0

    except Exception:
        return 0

#add_grn_overlay, add_success_overlay, and plot_grn_results are additional functions for video enhancement and results plotting
def add_grn_overlay(img, gene_expression, episode, total_episodes, reward, step, genome):
    try:
        from PIL import Image, ImageDraw, ImageFont

        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)

        try:
            font = ImageFont.truetype("arial.ttf", 14)
            small_font = ImageFont.truetype("arial.ttf", 10)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        # Create overlay
        overlay_height = 160
        overlay = Image.new('RGBA', (pil_img.width, overlay_height), (0, 0, 0, 200))
        pil_img.paste(overlay, (0, 0), overlay)

        # Episode info
        episode_text = f"GRN Episode {episode}/{total_episodes} | Step {step} | Reward: {reward:.2f}"
        draw.text((10, 5), episode_text, fill=(255, 255, 255), font=font)

        # Gene expression levels
        y_offset = 25
        active_genes = np.argsort(gene_expression)[-8:]

        colors = [
            (255, 100, 100), (100, 255, 100), (100, 150, 255), (255, 255, 100),
            (255, 150, 100), (150, 100, 255), (100, 255, 255), (255, 100, 255)
        ]

        for i, gene_idx in enumerate(active_genes):
            y_pos = y_offset + i*15
            expression_level = gene_expression[gene_idx]

            text = f"Gene {gene_idx:2d}: {expression_level:.3f}"
            color = colors[i % len(colors)]
            draw.text((10, y_pos), text, fill=color, font=small_font)

            # Expression bar
            bar_x, bar_y = 150, y_pos + 2
            bar_width, bar_height = 100, 10

            draw.rectangle([bar_x, bar_y, bar_x + bar_width, bar_y + bar_height],
                          fill=(40, 40, 40), outline=(80, 80, 80))

            expr_width = int(bar_width * np.clip(expression_level, 0, 1))
            draw.rectangle([bar_x, bar_y, bar_x + expr_width, bar_y + bar_height], fill=color)

        # Statistics
        stats_y = y_offset + 125
        active_genes_count = np.sum(gene_expression > 0.1)
        avg_expression = np.mean(gene_expression)
        max_expression = np.max(gene_expression)

        stats_text = f"Active: {active_genes_count}/{genome.num_genes} | Avg: {avg_expression:.3f} | Max: {max_expression:.3f}"
        draw.text((10, stats_y), stats_text, fill=(180, 180, 180), font=small_font)

        return np.array(pil_img)

    except Exception:
        return img

def add_success_overlay(img, episode, step, reward):
    try:
        from PIL import Image, ImageDraw, ImageFont

        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)

        success_overlay = Image.new('RGBA', (pil_img.width, 100), (0, 255, 0, 120))
        pil_img.paste(success_overlay, (0, pil_img.height//2 - 50), success_overlay)

        success_text = f"SUCCESS"
        reward_text = f"Episode {episode} | Step {step} | Reward: {reward:.2f}"

        try:
            big_font = ImageFont.truetype("arial.ttf", 24)
            small_font = ImageFont.truetype("arial.ttf", 16)
        except:
            big_font = None
            small_font = None

        text_y = pil_img.height//2 - 30
        draw.text((pil_img.width//2 - 100, text_y), success_text, fill=(255, 255, 255), font=big_font)
        draw.text((pil_img.width//2 - 120, text_y + 30), reward_text, fill=(255, 255, 255), font=small_font)

        return np.array(pil_img)

    except Exception:
        return img

def plot_grn_results(optimizer, test_results, genome, save_dir):
    plt.figure(figsize=(15, 10))

    # Evolution progress
    plt.subplot(2, 3, 1)
    generations = range(len(optimizer.fitness_history))
    plt.plot(generations, optimizer.fitness_history, 'b-', linewidth=2, label='Best Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Evolution Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Success rate evolution
    plt.subplot(2, 3, 2)
    success_pct = [s * 100 for s in optimizer.success_history]
    plt.plot(generations, success_pct, 'g-', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate Evolution')
    plt.grid(True, alpha=0.3)

    # Reward distribution
    plt.subplot(2, 3, 3)
    if test_results and 'all_rewards' in test_results:
        rewards = test_results['all_rewards']
        plt.hist(rewards, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.axvline(np.mean(rewards), color='red', linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
        plt.xlabel('Episode Reward')
        plt.ylabel('Frequency')
        plt.title('Reward Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Regulatory matrix heatmap
    plt.subplot(2, 3, 4)
    reg_matrix = genome.regulatory_matrix
    plt.imshow(reg_matrix, cmap='RdBu', aspect='auto', vmin=-0.3, vmax=0.3)
    plt.colorbar(label='Regulatory Strength')
    plt.xlabel('Target Gene')
    plt.ylabel('Source Gene')
    plt.title('Regulatory Matrix')

    # Gene parameters
    plt.subplot(2, 3, 5)
    plt.hist(genome.activation_thresholds, bins=15, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Activation Threshold')
    plt.ylabel('Number of Genes')
    plt.title('Gene Activation Thresholds')
    plt.grid(True, alpha=0.3)

    # Network connectivity
    plt.subplot(2, 3, 6)
    connections_per_gene = np.sum(np.abs(reg_matrix) > 0.01, axis=1)
    plt.bar(range(len(connections_per_gene)), connections_per_gene, alpha=0.7, color='cyan')
    plt.xlabel('Gene Index')
    plt.ylabel('Connections')
    plt.title('Gene Connectivity')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/grn_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_grn_results(genome, optimizer, test_results, save_dir):
    with open(f"{save_dir}/best_grn_genome.pkl", 'wb') as f:
        pickle.dump(genome, f)

    with open(f"{save_dir}/test_results.pkl", 'wb') as f:
        pickle.dump(test_results, f)

    evolution_data = {
        'fitness_history': optimizer.fitness_history,
        'success_history': optimizer.success_history,
        'best_genomes': optimizer.best_genome_history,
        'population_size': optimizer.population_size,
        'mutation_rate': optimizer.mutation_rate
    }

    with open(f"{save_dir}/evolution_history.pkl", 'wb') as f:
        pickle.dump(evolution_data, f)

    summary = {
        'experiment_type': 'Gene Regulatory Network Robot Control',
        'timestamp': datetime.now().isoformat(),
        'final_performance': {
            'success_rate': test_results['success_rate'],
            'avg_reward': test_results['avg_reward'],
            'std_reward': test_results['std_reward'],
            'total_successes': test_results['total_successes'],
            'total_episodes': len(test_results['all_rewards'])
        },
        'grn_architecture': {
            'num_genes': genome.num_genes,
            'num_actuators': genome.num_actuators,
            'sensor_dims': genome.sensor_dims,
            'regulatory_connections': int(np.sum(np.abs(genome.regulatory_matrix) > 0.01)),
            'sparsity': float(np.sum(np.abs(genome.regulatory_matrix) > 0.01) / (genome.num_genes ** 2))
        },
        'evolution_stats': {
            'generations': len(optimizer.fitness_history),
            'population_size': optimizer.population_size,
            'best_fitness': genome.fitness,
            'final_success_rate': optimizer.success_history[-1] if optimizer.success_history else 0
        }
    }

    with open(f"{save_dir}/experiment_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

def run_grn_experiment(population_size=16, max_generations=20, test_episodes=50, create_video=True):

    # Set seeds for reproducibility
    # The best results over 25 different seeds where achieved with seed = 16
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Create environment
    try:
        from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
        import pybullet as p

        env = KukaDiverseObjectEnv(
            renders=False,
            isDiscrete=False,
            removeHeightHack=False,
            maxSteps=30
        )
        env.cid = p.connect(p.DIRECT)

    except Exception as e:
        print(f"Environment creation failed: {e}")
        return None

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results/grn_experiment_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    try:
        # Initialize optimizer
        optimizer = GRNEvolutionaryOptimizer(
            population_size=population_size,
            elite_fraction=0.25,
            mutation_rate=0.2
        )

        # Evolution phase
        print("Starting GRN evolution..")
        start_time = time.time()
        best_genome = optimizer.evolve(env, max_generations=max_generations)
        evolution_time = time.time() - start_time

        print(f"Evolution completed in {evolution_time:.1f}s")
        print(f"Best genome: Fitness={best_genome.fitness:.2f}, Success={best_genome.success_rate:.1%}")

        # Testing phase
        print("Testing evolved GRN..")
        test_results = test_grn_genome(env, best_genome, episodes=test_episodes)

        print(f"Final results:")
        print(f"  Success rate: {test_results['success_rate']:.1%}")
        print(f"  Average reward: {test_results['avg_reward']:.2f}")

        # Create video
        video_created = False
        if create_video:
            print("Creating demonstration video..")
            video_path = os.path.join(save_dir, "grn_demo.mp4")
            frames = create_grn_video(env, best_genome, video_path, episodes=3)
            video_created = frames > 0

        # Analysis and saving
        print("Creating analysis plots...")
        plot_grn_results(optimizer, test_results, best_genome, save_dir)

        print("Saving results...")
        save_grn_results(best_genome, optimizer, test_results, save_dir)

        print(f"Results saved to: {save_dir}")

        return {
            'success': True,
            'best_genome': best_genome,
            'test_results': test_results,
            'save_dir': save_dir,
            'video_created': video_created
        }

    except Exception as e:
        print(f"Experiment failed: {e}")
        return {'success': False, 'error': str(e)}

    finally:
        try:
            env.close()
            p.disconnect()
        except:
            pass

if __name__ == "__main__":
    print("Gene Regulatory Network for Robotic Grasping")
    print("=" * 50)

    # Population size and generations number (Expiremently, small sizes gave robust results)
    results = run_grn_experiment(
        population_size=12,
        max_generations=15,
        test_episodes=40,
        create_video=True
    )

    if results['success']:
        print(f"\nExperiment completed successfully")
        print(f"Success rate: {results['test_results']['success_rate']:.1%}")
    else:
        print(f"Experiment failed: {results['error']}")
    

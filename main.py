import yaml
import logging
import os
from src.data.generate_data import generate_microwave_data
from src.visualization.eda_plots import run_eda
from src.features.build_features import build_features
from src.models.train_model import train_and_evaluate

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Nokia TLC Network Anomaly Detection Pipeline")
    
    # Load config
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # 1. Data Generation
    logger.info("Step 1: Generating synthetic data...")
    df_raw = generate_microwave_data(
        n_samples=config['data']['samples'], 
        seed=config['data']['seed']
    )
    os.makedirs('data/raw', exist_ok=True)
    df_raw.to_csv(config['data']['raw_path'], index=False)
    
    # 2. EDA
    logger.info("Step 2: Performing Exploratory Data Analysis...")
    run_eda(config['data']['raw_path'])
    
    # 3. Feature Engineering
    logger.info("Step 3: Building features...")
    build_features(
        input_path=config['data']['raw_path'], 
        output_path=config['data']['processed_path']
    )
    
    # 4. Model Training & Evaluation
    logger.info("Step 4: Training and evaluating models...")
    metrics = train_and_evaluate(config['data']['processed_path'])
    
    logger.info("Pipeline completed successfully!")
    logger.info(f"Model Results:\n{metrics}")

if __name__ == "__main__":
    main()

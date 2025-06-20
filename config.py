import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"
PRODUCT_DESCRIPTION = "The Autonomous Brainy is a high-performance AI workstation series powered by up to 8 NVIDIA RTX 4090 GPUs, designed to run models with up to 70 billion parameters locally. Ideal for developers, researchers, and AI labs, each configuration offers powerful FP32 compute up to 1.32 petaflops. Available models: (1) Brainy with 1 x RTX 4090 – entry-level AI rig for local inference ($5000 USD), (2) Brainy with 2 x RTX 4090 – fine-tuning and scalable workloads ($9000 USD), (3) Brainy with 4 x RTX 4090 – professional-grade model training ($19000 USD), (4) Brainy with 6 x RTX 4090 – robust multi-model pipelines ($25000 USD), and (5) Brainy with 8 x RTX 4090 – full-scale AI research and deployment ($32000 USD). All configurations are in stock and deliver extreme local performance without relying on the cloud. Learn more at https://www.autonomous.ai/robots/brainy"

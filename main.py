import argparse
from detector.video_utils import process_video

def parse_args():
    parser = argparse.ArgumentParser(description="Parking Spot Detector")
    parser.add_argument("--video", required=True, help="Caminho para o vídeo de entrada")
    parser.add_argument("--config", default="config.py", help="Arquivo de configuração de ROIs")
    return parser.parse_args()

def main():
    args = parse_args()
    process_video(video_path=args.video, config_path=args.config)

if __name__ == "__main__":
    main()
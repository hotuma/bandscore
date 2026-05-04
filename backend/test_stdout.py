import sys
import logging

# stdoutバッファリング無効化
sys.stdout.reconfigure(line_buffering=True)

# ルートロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

print("[PRINT] This is a print statement")
logger.info("[LOGGER] This is a logger info statement")
logger.debug("[LOGGER] This is a logger debug statement")
print("[PRINT] Test completed")

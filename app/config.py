from pathlib import Path


PROJECT_NAME = "Reconhecimento de Objetos com YOLO"
WINDOW_NAME = "Reconhecimento de Objetos com YOLO e OpenCV"

DEFAULT_MODEL = "yolo11n.pt"
FALLBACK_MODEL = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.35
DEFAULT_SOURCE = "0"

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
PRINTS_DIR = OUTPUTS_DIR / "prints"
VIDEOS_DIR = OUTPUTS_DIR / "videos"
LOGS_DIR = OUTPUTS_DIR / "logs"

DEFAULT_CLASSES = [
    "person",
    "bottle",
    "cell phone",
    "laptop",
    "book",
    "cup",
    "chair",
    "keyboard",
    "mouse",
    "backpack",
    "remote",
]

CLASS_TRANSLATIONS = {
    "person": "pessoa",
    "bicycle": "bicicleta",
    "car": "carro",
    "motorcycle": "moto",
    "airplane": "aviao",
    "bus": "onibus",
    "train": "trem",
    "truck": "caminhao",
    "boat": "barco",
    "traffic light": "semaforo",
    "fire hydrant": "hidrante",
    "stop sign": "placa pare",
    "parking meter": "parquimetro",
    "bench": "banco",
    "bird": "passaro",
    "cat": "gato",
    "dog": "cachorro",
    "horse": "cavalo",
    "sheep": "ovelha",
    "cow": "vaca",
    "elephant": "elefante",
    "bear": "urso",
    "zebra": "zebra",
    "giraffe": "girafa",
    "backpack": "mochila",
    "umbrella": "guarda-chuva",
    "handbag": "bolsa",
    "tie": "gravata",
    "suitcase": "mala",
    "frisbee": "frisbee",
    "skis": "esquis",
    "snowboard": "snowboard",
    "sports ball": "bola",
    "kite": "pipa",
    "baseball bat": "taco de baseball",
    "baseball glove": "luva de baseball",
    "skateboard": "skate",
    "surfboard": "prancha",
    "tennis racket": "raquete",
    "bottle": "garrafa",
    "wine glass": "taca",
    "cup": "copo",
    "fork": "garfo",
    "knife": "faca",
    "spoon": "colher",
    "bowl": "tigela",
    "banana": "banana",
    "apple": "maca",
    "sandwich": "sanduiche",
    "orange": "laranja",
    "broccoli": "brocolis",
    "carrot": "cenoura",
    "hot dog": "cachorro-quente",
    "pizza": "pizza",
    "donut": "rosquinha",
    "cake": "bolo",
    "chair": "cadeira",
    "couch": "sofa",
    "potted plant": "planta",
    "bed": "cama",
    "dining table": "mesa",
    "toilet": "vaso sanitario",
    "tv": "televisao",
    "laptop": "notebook",
    "mouse": "mouse",
    "remote": "controle remoto",
    "keyboard": "teclado",
    "cell phone": "celular",
    "microwave": "micro-ondas",
    "oven": "forno",
    "toaster": "torradeira",
    "sink": "pia",
    "refrigerator": "geladeira",
    "book": "livro",
    "clock": "relogio",
    "vase": "vaso",
    "scissors": "tesoura",
    "teddy bear": "urso de pelucia",
    "hair drier": "secador",
    "toothbrush": "escova de dente",
}

DASHBOARD_COLORS = {
    "panel": (18, 24, 31),
    "panel_secondary": (32, 42, 52),
    "text": (245, 248, 250),
    "muted": (185, 196, 205),
    "success": (90, 220, 150),
    "warning": (0, 190, 255),
    "danger": (80, 90, 255),
    "dark": (10, 12, 16),
}

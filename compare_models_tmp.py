import torch
from pathlib import Path
from train_models.model_microscopio import train_model_microscopio as t

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_, _, test_loader = t.get_data_loaders()
criterion = torch.nn.CrossEntropyLoss()
base = Path('train_models/model_microscopio/resultados')

for name in ['ConvNeXt-Tiny', 'DenseNet-121']:
    p = base / name / 'best.pt'
    if not p.exists():
        print(f'{name}: NO_WEIGHTS')
        continue
    cfg = next(c for c in t.COMPETITION_MODELS if c.name == name)
    model = t.build_model(cfg, num_classes=len(t.CLASS_NAMES), device=device)
    model.load_state_dict(torch.load(p, map_location=device, weights_only=True))
    r = t.evaluate(model, test_loader, criterion, device)
    print(f"{name}: acc={r['accuracy']:.4f} recall={r['recall']:.4f} precision={r['precision']:.4f} f1={r['f1']:.4f} FN={r['fn']} FP={r['fp']}")

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
r = json.loads((ROOT/'results/metrics.json').read_text())
data = np.load(ROOT/'results/visual_data.npz')
train = np.loadtxt(ROOT/'data/optdigits.tra', delimiter=',')[:, :64]
font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 26)
small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 23)


def tile(image, row, col, pixels, label, detail):
    x, y = 30 + col*280, 20 + row*310
    p = Image.fromarray(np.uint8(np.clip(pixels, 0, 1)*255))
    p = p.resize((192, 192), Image.Resampling.NEAREST).convert('RGB')
    image.paste(p, (x+32, y))
    d = ImageDraw.Draw(image)
    d.text((x+20, y+204), label, fill='black', font=font)
    d.text((x+20, y+240), detail, fill='black', font=small)


im = Image.new('RGB', (1150, 300), 'white')
tile(im, 0, 0, data['images'][0], 'Query row 1', 'True 0   Pred 0')
for j, row in enumerate(r['demo']['neighbor_rows_1based']):
    tile(im, 0, j+1, train[row-1].reshape(8, 8)/16,
         f'Neighbor {j+1}', f'Label 0   Row {row}')
im.save(ROOT/'demo_neighbors.png', dpi=(220, 220))

im = Image.new('RGB', (870, 620), 'white')
for n, i in enumerate(data['errors'][:6]):
    tile(im, n//3, n%3, data['images'][i],
         f'Test row {i+1}', f"True {data['true'][i]}   Pred {data['pred'][i]}")
im.save(ROOT/'error_samples.png', dpi=(220, 220))

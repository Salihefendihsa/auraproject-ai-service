"""Test composite card rendering."""
import sys
sys.path.insert(0, 'ai_service')

from outfit_recommender import generate_outfit_combos
from outfit_renderer import render_all_outfit_cards
import shutil
from pathlib import Path

# Test: User has beige pants, want recommendations
fixed = {'name': 'Beige Chinos', 'color': 'beige', 'image': 'bottom_002_beige_chinos.jpg'}
outfits = generate_outfit_combos(fixed, 'bottom', num_outfits=3, gender='male')

print(f'Generated {len(outfits)} outfits')

# Check if image fields are present
for outfit in outfits:
    rank = outfit["rank"]
    print(f'Outfit {rank}:')
    for cat, item in outfit['items'].items():
        img = item.get('image', 'NO IMAGE')
        print(f'  {cat}: {img}')

# Render cards
output_dir = 'test_composite_cards'
results = render_all_outfit_cards(outfits, output_dir, 'male')
print(f'Rendered {len(results)} cards to {output_dir}/')

# Copy first card to artifacts for review
card_path = Path(f'{output_dir}/outfit_1.png')
if card_path.exists():
    shutil.copy(str(card_path), 'C:/Users/SALİH/.gemini/antigravity/brain/29258f29-9f87-4825-af04-11f4efc9e33d/composite_card_test.png')
    print('Copied to artifacts for review')

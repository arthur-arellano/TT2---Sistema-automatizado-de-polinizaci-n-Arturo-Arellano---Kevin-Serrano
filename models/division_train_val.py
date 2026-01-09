import os
import random
import shutil

# Configura rutas
base_dir = r"C:\Users\artur\Downloads\project-7-at-2025-10-07-00-13-cb813027"
img_dir = os.path.join(base_dir, 'images')
label_dir = os.path.join(base_dir, 'labels')

# Carpetas destino
splits = ['train', 'val', 'test']
split_ratio = [0.7, 0.2, 0.1]  # train, val, test

for split in splits:
    os.makedirs(os.path.join(base_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'labels', split), exist_ok=True)

# Obtener y mezclar las imágenes
images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
random.shuffle(images)

n_total = len(images)
n_train = int(split_ratio[0] * n_total)
n_val = int(split_ratio[1] * n_total)
n_test = n_total - n_train - n_val

splits_counts = {
    'train': images[:n_train],
    'val': images[n_train:n_train + n_val],
    'test': images[n_train + n_val:]
}

# Copiar archivos
for split, split_images in splits_counts.items():
    for img_name in split_images:
        label_name = img_name.rsplit('.', 1)[0] + '.txt'

        # Imagen
        shutil.copy(os.path.join(img_dir, img_name),
                    os.path.join(base_dir, 'images', split, img_name))

        # Etiqueta
        shutil.copy(os.path.join(label_dir, label_name),
                    os.path.join(base_dir, 'labels', split, label_name))

print("✅ Dataset dividido exitosamente.")

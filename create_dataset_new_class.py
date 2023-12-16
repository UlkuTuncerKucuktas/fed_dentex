import os
import shutil
import random
import argparse
import matplotlib.pyplot as plt
from collections import Counter

def parse_arguments():
    parser = argparse.ArgumentParser(description='Split dataset into training and validation sets and analyze class distribution.')
    parser.add_argument('--image_folder', type=str, required=True, help='Path to the folder containing images')
    parser.add_argument('--label_folder', type=str, required=True, help='Path to the folder containing labels')
    parser.add_argument('--dest_base_path', type=str, required=True, help='Base path for the destination folders')
    parser.add_argument('--train_split', type=float, default=0.9, help='Split ratio for training set')
    parser.add_argument('--target_class', type=int, required=True, help='Target class for the first split')
    parser.add_argument('--additional_splits', nargs='*', type=float, default=[], help='Additional split ratios for remaining training set')
    return parser.parse_args()


def analyze_split(label_folder):
    class_counts, image_count = parse_labels(label_folder)
    return class_counts, image_count

def create_split(source_image_folder, source_label_folder, dest_image_folder, dest_label_folder, num_to_select):
    image_files = os.listdir(source_image_folder)

    selected_image_files = random.sample(image_files, min(num_to_select, len(image_files)))

    if not os.path.exists(dest_image_folder):
        os.makedirs(dest_image_folder)
    if not os.path.exists(dest_label_folder):
        os.makedirs(dest_label_folder)

    for image_file in selected_image_files:
        label_file = image_file.replace('.png', '.txt')
        
        shutil.move(os.path.join(source_image_folder, image_file), os.path.join(dest_image_folder, image_file))
        if os.path.exists(os.path.join(source_label_folder, label_file)):
            shutil.move(os.path.join(source_label_folder, label_file), os.path.join(dest_label_folder, label_file))


def create_split_selected(source_folder, dest_folder, selected_files):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for file in selected_files:
        try:
            shutil.move(os.path.join(source_folder, file), os.path.join(dest_folder, file))
        except:
            print(os.path.join(source_folder, file))

def parse_labels(label_folder):
    class_counts = Counter()
    for label_file in os.listdir(label_folder):
        with open(os.path.join(label_folder, label_file), 'r') as file:
            for line in file:
                class_id = int(line.split()[0])
                class_counts[class_id] += 1
    return class_counts, len(os.listdir(label_folder))

def plot_class_distribution(class_counts, title, save_path, color_map):
    labels = list(class_counts.keys())
    sizes = list(class_counts.values())
    colors = [color_map.get(label, '#333333') for label in labels]  # Default to a dark gray color if not found

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def plot_image_counts(image_counts, title, save_path):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(image_counts)), list(image_counts.values()), tick_label=list(image_counts.keys()))
    plt.xlabel('Split')
    plt.ylabel('Number of Images')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def find_images_with_label(label_folder, label_to_find):
    images_with_label = set()
    for label_file in os.listdir(label_folder):
        with open(os.path.join(label_folder, label_file), 'r') as file:
            for line in file:
                class_id = int(line.split()[0])
                if class_id == label_to_find:
                    image_file = label_file.replace('.txt', '.png')
                    images_with_label.add(image_file)
                    break
    return images_with_label



def main():
    args = parse_arguments()

    # Initial setup
    image_folder = args.image_folder
    label_folder = args.label_folder
    dest_base_path = args.dest_base_path
    total_images = len(os.listdir(image_folder))
    train_count = int(total_images * args.train_split)
    val_count = total_images - train_count

    # Initial splitting into train and validation sets
    train_image_folder = os.path.join(dest_base_path, 'train/images')
    train_label_folder = os.path.join(dest_base_path, 'train/labels')
    val_image_folder = os.path.join(dest_base_path, 'val/images')
    val_label_folder = os.path.join(dest_base_path, 'val/labels')
    create_split(image_folder, label_folder, train_image_folder, train_label_folder, train_count)
    create_split(image_folder, label_folder, val_image_folder, val_label_folder, val_count)

    # Copy training set to a temporary location
    temp_train_image_folder = os.path.join(dest_base_path, 'temp_train/images')
    temp_train_label_folder = os.path.join(dest_base_path, 'temp_train/labels')
    shutil.copytree(train_image_folder, temp_train_image_folder)
    shutil.copytree(train_label_folder, temp_train_label_folder)

    # Move images for the first split
    target_images = find_images_with_label(temp_train_label_folder, args.target_class)
    first_split_image_folder = os.path.join(dest_base_path, 'split_1/images')
    first_split_label_folder = os.path.join(dest_base_path, 'split_1/labels')
    create_split_selected(temp_train_image_folder, first_split_image_folder, target_images)
    create_split_selected(temp_train_label_folder, first_split_label_folder, {f.replace('.png', '.txt') for f in target_images})

    # Calculate split sizes
    remaining_train_images_count = len(os.listdir(temp_train_image_folder))
    split_sizes = [int(remaining_train_images_count * ratio) for ratio in args.additional_splits]
    split_sizes[-1] += remaining_train_images_count - sum(split_sizes)  # Adjust the last split

    # Create additional splits
    for i, split_size in enumerate(split_sizes):
        current_train_images = os.listdir(temp_train_image_folder)
        selected_images = set(random.sample(current_train_images, split_size))
        selected_labels = {f.replace('.png', '.txt') for f in selected_images}

        split_image_folder = os.path.join(dest_base_path, f'train_{int(args.additional_splits[i] * 100)}/images')
        split_label_folder = os.path.join(dest_base_path, f'train_{int(args.additional_splits[i] * 100)}/labels')
        create_split_selected(temp_train_image_folder, split_image_folder, selected_images)
        create_split_selected(temp_train_label_folder, split_label_folder, selected_labels)

    # Remove temporary training set
    shutil.rmtree(os.path.join(dest_base_path, 'temp_train'))
    splits_info = {}
    image_counts = {}

    # Analyze main train and validation sets
    train_class_counts, train_image_count = parse_labels(train_label_folder)
    val_class_counts, val_image_count = parse_labels(val_label_folder)

    splits_info['train'] = train_class_counts
    splits_info['val'] = val_class_counts
    image_counts['train'] = train_image_count
    image_counts['val'] = val_image_count

    # Analyze the first split (split_1)
    first_split_label_folder = os.path.join(dest_base_path, 'split_1/labels')
    split_1_class_counts, split_1_image_count = parse_labels(first_split_label_folder)
    splits_info['split_1'] = split_1_class_counts
    image_counts['split_1'] = split_1_image_count

    # Analyze additional splits
    for split_ratio in args.additional_splits:
        split_name = f'train_{int(split_ratio * 100)}'
        additional_train_label_folder = os.path.join(dest_base_path, f'{split_name}/labels')
        split_class_counts, split_image_count = parse_labels(additional_train_label_folder)

        splits_info[split_name] = split_class_counts
        image_counts[split_name] = split_image_count

    # Define color map for class distribution plots
    color_map = {
        0: 'gold', 
        1: 'yellowgreen', 
        2: 'lightcoral', 
        3: 'lightskyblue'
    }

    # Plotting class distribution for each split
    for split, class_counts in splits_info.items():
        plot_class_distribution(class_counts, f'{split} Class Distribution', os.path.join(dest_base_path, f'{split}_class_distribution.png'), color_map)

    # Plotting image counts in each split
    plot_image_counts(image_counts, 'Image Counts in Each Split', os.path.join(dest_base_path, 'split_image_counts.png'))

if __name__ == "__main__":
    main()
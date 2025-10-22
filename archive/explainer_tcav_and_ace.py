import os, glob
import random
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils import age_to_group
from scipy.stats import ttest_ind

from skimage.segmentation import slic
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import torch
from torch.utils.data import IterableDataset, DataLoader

import torchvision
import torchvision.models as models
from torchvision import transforms
from torchvision.utils import make_grid

from captum.attr import LayerGradientXActivation, LayerIntegratedGradients
from captum.concept import TCAV
from captum.concept import Concept
from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset
from captum.concept._utils.common import concepts_to_str

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

############################################
################### TCAV ###################
############################################


os.makedirs('./exp_output/tcav_and_ace_plots', exist_ok=True)

def transform(img):
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )(img)


def get_tensor_from_filename(filename):
    img = Image.open(filename).convert("RGB")
    return transform(img)


def load_image_tensors_multi_dir(class_dirs, transform_func=None):
    all_tensors = []
    all_filenames = []
    
    for class_dir in class_dirs:
        filenames = glob.glob(os.path.join(class_dir, '*.jpg')) + \
                   glob.glob(os.path.join(class_dir, '*.png'))
        filenames.sort()  # Sort for consistent ordering
        
        for filename in filenames:
            img = Image.open(filename).convert('RGB')
            if transform_func:
                tensor = transform_func(img)
            else:
                tensor = img
            all_tensors.append(tensor)
            all_filenames.append(filename)

    return all_tensors, all_filenames


def assemble_concept(name, id, concepts_path="tcav_data/concept/"):
    concept_path = os.path.join(concepts_path, name) + "/"
    dataset = CustomIterableDataset(get_tensor_from_filename, concept_path)
    concept_iter = dataset_to_dataloader(dataset)

    return Concept(id=id, name=name, data_iter=concept_iter)


concepts_path = "./tcav_data/concept"

beard_concept = assemble_concept("beard", 0, concepts_path=concepts_path)
glasses_concept = assemble_concept("glasses", 1, concepts_path=concepts_path)
wrinkle_concept = assemble_concept("wrinkle", 2, concepts_path=concepts_path)

random_0_concept = assemble_concept("random_0", 3, concepts_path=concepts_path)
random_1_concept = assemble_concept("random_1", 4, concepts_path=concepts_path)


n_figs = 5
n_concepts = 5

fig, axs = plt.subplots(n_concepts, n_figs + 1, figsize = (25, 4 * n_concepts))

for c, concept in enumerate([beard_concept, glasses_concept, wrinkle_concept, random_0_concept, random_1_concept]):
    concept_path = os.path.join(concepts_path, concept.name) + "/"
    img_files = glob.glob(concept_path + '*')
    for i, img_file in enumerate(img_files[:n_figs + 1]):
        if os.path.isfile(img_file):
            if i == 0:
                axs[c, i].text(1.0, 0.5, str(concept.name), ha='right', va='center', family='sans-serif', size=24)
            else:
                img = plt.imread(img_file)
                axs[c, i].imshow(img)

            axs[c, i].axis('off')

plt.tight_layout()
plt.savefig('./exp_output/tcav_and_ace_plots/concept_examples.png', dpi=300, bbox_inches='tight')
print("Saved concept examples plot to ./exp_output/tcav_and_ace_plots/concept_examples.png")
plt.close()


# Load model
def load_utkface_model(model_path='train_output/best_resnet18_gender.pth', num_classes=2):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)    
    state = torch.load(model_path, map_location='cpu')
    state_dict = state.get("model_state", state) if isinstance(state, dict) else state
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_utkface_model()
print("Loaded trained UTKFace model")


# Layers for TCAV
layers = ['layer3.0.conv2', 'layer4.1.conv1', 'layer3.1.conv2']

# def print_model_layers(model):
#     for name, module in model.named_modules():
#         if len(list(module.children())) == 0:
#             print(f"Layer name: {name}, Module: {type(module).__name__}")

# print_model_layers(model)



mytcav = TCAV(model=model,
              layers=layers,
              layer_attr_method = LayerIntegratedGradients(
                model, None, multiply_by_inputs=False))


experimental_set_rand = [[beard_concept, random_0_concept], [beard_concept, random_1_concept]]


# Set random seeds for reproducible results
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Load images from folders
data_dirs = [
    './train_data/part1_cropped',
    './train_data/part2_cropped', 
    './train_data/part3_cropped'

]
face_imgs, _ = load_image_tensors_multi_dir(data_dirs, transform_func=None)
print(f"Loaded {len(face_imgs)} face images total")

# Use 100 images
max_images = 100
face_imgs = face_imgs[:max_images]


fig, axs = plt.subplots(1, 5, figsize = (25, 5))
axs[0].imshow(face_imgs[40])
axs[1].imshow(face_imgs[41])
axs[2].imshow(face_imgs[34])
axs[3].imshow(face_imgs[31])
axs[4].imshow(face_imgs[30])

axs[0].axis('off')
axs[1].axis('off')
axs[2].axis('off')
axs[3].axis('off')
axs[4].axis('off')

plt.tight_layout()
plt.savefig('./exp_output/tcav_and_ace_plots/face_samples.png', dpi=300, bbox_inches='tight')
print("Saved face samples plot to ./exp_output/tcav_and_ace_plots/face_samples.png")
plt.close()


# Load sample images from folder
face_tensors = torch.stack([transform(img) for img in face_imgs])
print(f"Created face tensor stack with shape: {face_tensors.shape} on device: {face_tensors.device}")


# Gender class indices
GENDER_MALE = 0
GENDER_FEMALE = 1

target_class = GENDER_MALE
print(f"Target class set to: {target_class} (GENDER_MALE)")


print("Starting TCAV interpretation for random experimental sets")
tcav_scores_w_random = mytcav.interpret(inputs=face_tensors,
                                        experimental_sets=experimental_set_rand,
                                        target=target_class,
                                        n_steps=5,
                                       )
print("TCAV interpretation for random sets completed!")



def format_float(f):
    return float('{:.3f}'.format(f) if abs(f) >= 0.0005 else '{:.3e}'.format(f))

def plot_tcav_scores(experimental_sets, tcav_scores, save_name):
    fig, ax = plt.subplots(1, len(experimental_sets), figsize = (25, 7))

    barWidth = 1 / (len(experimental_sets[0]) + 1)

    for idx_es, concepts in enumerate(experimental_sets):

        concepts = experimental_sets[idx_es]
        concepts_key = concepts_to_str(concepts)

        pos = [np.arange(len(layers))]
        for i in range(1, len(concepts)):
            pos.append([(x + barWidth) for x in pos[i-1]])
        _ax = (ax[idx_es] if len(experimental_sets) > 1 else ax)
        for i in range(len(concepts)):
            val = [format_float(scores['sign_count'][i]) for layer, scores in tcav_scores[concepts_key].items()]
            _ax.bar(pos[i], val, width=barWidth, edgecolor='white', label=concepts[i].name)

        _ax.set_xlabel('Set {}'.format(str(idx_es)), fontweight='bold', fontsize=16)
        _ax.set_xticks([r + 0.3 * barWidth for r in range(len(layers))])
        _ax.set_xticklabels(layers, fontsize=16)
        _ax.legend(fontsize=16)

    plt.tight_layout()
    plt.savefig(f'./exp_output/tcav_and_ace_plots/{save_name}.png', dpi=300, bbox_inches='tight')
    print(f"Saved TCAV scores plot to ./exp_output/tcav_and_ace_plots/{save_name}.png")
    plt.close()


plot_tcav_scores(experimental_set_rand, tcav_scores_w_random, 'tcav_scores_random')


experimental_set_face_attributes = [[beard_concept, glasses_concept, wrinkle_concept]]


print("Starting TCAV interpretation for face attributes")
tcav_scores_w_zig_dot = mytcav.interpret(inputs=face_tensors,
                                         experimental_sets=experimental_set_face_attributes,
                                         target=target_class,
                                         n_steps=5)
print("TCAV interpretation for face attributes completed!")


plot_tcav_scores(experimental_set_face_attributes, tcav_scores_w_zig_dot, 'tcav_scores_face_attributes')



##########################################
# Subgroup-specific TCAV
##########################################

def clean_tensor_string(val):
    """Parse tensor(X) format from CSV back to integer"""
    if isinstance(val, str) and val.startswith('tensor(') and val.endswith(')'):
        return int(val[7:-1])  # Extract X from tensor(X)
    return int(float(val)) if val else -1

def load_test_predictions(csv_path):
    rows = []
    if not os.path.exists(csv_path):
        print(f"Test predictions CSV not found: {csv_path}")
        return rows
    with open(csv_path, newline='') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            try:
                r['age'] = clean_tensor_string(r.get('age', ''))
            except Exception:
                r['age'] = -1
            try:
                r['race'] = clean_tensor_string(r.get('race', ''))
            except Exception:
                r['race'] = -1
            try:
                r['gender_true'] = int(float(r.get('gender_true', '') or -1))
            except Exception:
                r['gender_true'] = -1
            try:
                r['pred_gender'] = int(float(r.get('pred_gender', '') or -1))
            except Exception:
                r['pred_gender'] = -1
            # compute age_group using project's util
            r['age_group'] = age_to_group(r['age']) if r['age'] >= 0 else None
            rows.append(r)
    return rows


def get_subgroup_tensors(rows, transform_func, age_group=None, race=None, gender=None, max_images=200):
    filtered = [r for r in rows if (age_group is None or r.get('age_group') == age_group)
                and (race is None or r.get('race') == race)
                and (gender is None or r.get('gender_true') == gender)]
    imgs = []
    file_meta = []
    for r in filtered[:max_images]:
        fp = r.get('filepath') or r.get('file')
        if not fp:
            continue
        # Fix path if it uses old /data/ prefix instead of train_data/
        if '/data/' in fp:
            fp = fp.replace('/data/', '/train_data/')
        if not os.path.exists(fp):
            continue
        try:
            img = Image.open(fp).convert('RGB')
            imgs.append(transform_func(img))
            file_meta.append(r)
        except Exception as e:
            continue
    if not imgs:
        return None, file_meta
    return torch.stack(imgs), file_meta


def run_tcav_on_subgroups(mytcav, experimental_sets, target, test_csv_path, transform_func, subgroup_requests):
    rows = load_test_predictions(test_csv_path)
    for sg in subgroup_requests:
        age_g = sg.get('age_group')
        race_g = sg.get('race')
        gender_g = sg.get('gender')

        tensors, meta = get_subgroup_tensors(rows, transform_func, age_group=age_g, race=race_g, gender=gender_g)
        sg_name = f"age_{age_g or 'all'}__race_{race_g if race_g is not None else 'all'}__gender_{gender_g if gender_g is not None else 'all'}"
        if tensors is None or tensors.size(0) < 8:
            print(f"Skipping TCAV for subgroup {sg_name}: not enough images ({0 if tensors is None else tensors.size(0)})")
            continue

        print(f"Running TCAV on subgroup {sg_name} with {tensors.size(0)} images")
        try:
            tcav_scores = mytcav.interpret(inputs=tensors,
                                           experimental_sets=experimental_sets,
                                           target=target,
                                           n_steps=5)
            save_name = f"tcav_scores_face_attributes_subgroup_{sg_name}"
            plot_tcav_scores(experimental_sets, tcav_scores, save_name)
        except Exception as e:
            print(f"TCAV failed for subgroup {sg_name}: {e}")


# Example subgroup requests for Team A (gender classifier) -- adjust as needed
test_csv_path = './train_output/test_predictions.csv'
subgroup_requests = [
    {"age_group": None, "race": None, "gender": None},
    {"age_group": 'young_adult', "race": 3, "gender": None},
    {"age_group": 'child', "race": 1, "gender": None},
]

# Run subgroup TCAV (reuse the face attribute experimental set and target defined earlier)
try:
    run_tcav_on_subgroups(mytcav, experimental_set_face_attributes, target_class, test_csv_path, transform, subgroup_requests)
except Exception as e:
    print("Subgroup TCAV run encountered an error:", e)


###########################################
################### ACE ###################
###########################################

model = model.to(DEVICE)
print(f"Model moved to {DEVICE} for ACE")


DATA_ROOT = Path("./")
CONCEPTS_PATH = Path("./tcav_data/concept")
FACE_DATA_PATHS = [
    "./train_data/part1_cropped",
    "./train_data/part2_cropped", 
    "./train_data/part3_cropped"
]

CONCEPTS_PATH.mkdir(parents=True, exist_ok=True)


IMG_SIZE = 224 
ACE_LAYER = None
for name, module in model.named_modules():
    if name == 'layer4.1.conv2':
        ACE_LAYER = module
        break

def segment_image(img_pil, n_segments=40, compactness=10):
    arr = np.array(img_pil)
    seg = slic(arr, n_segments=n_segments, compactness=compactness,
               channel_axis=2, start_label=0)
    return seg  # HxW labels

_layer_buf = []

def _ace_hook(m, i, o):
    _layer_buf.append(o.detach().cpu())

# Register the hook once
hook = ACE_LAYER.register_forward_hook(_ace_hook)

def forward_and_get_acts(batch_tensor):

    _layer_buf.clear()
    with torch.no_grad():
        _ = model(batch_tensor.to(DEVICE))
    return _layer_buf[0]

def extract_patch_embeddings(img_pil, tensor, segments):

    # [B=1, C, Hf, Wf] -> [C, Hf, Wf]
    acts = forward_and_get_acts(tensor.unsqueeze(0))[0]
    C, Hf, Wf = acts.shape

    # Resize integer labels to feature-map size (nearest keeps ids intact)
    seg_small = np.array(
        Image.fromarray(segments.astype(np.int32)).resize(
            (Wf, Hf), resample=Image.NEAREST
        )
    )

    vecs, masks = [], []
    MIN_PIXELS = 1  # less restrictive for small feature maps
    for seg_id in np.unique(seg_small):
        mask = (seg_small == seg_id)            # [Hf, Wf] bool
        if mask.sum() < MIN_PIXELS:
            continue
        # Mean over spatial positions where mask is True -> [C]
        v = acts[:, mask].mean(dim=1).numpy()
        vecs.append(v)
        masks.append(mask)

    if vecs:
        vecs = np.stack(vecs)                   # [num_patches, C]
    else:
        vecs = np.zeros((0, C), dtype=np.float32)

    return vecs, masks


def cluster_embeddings(all_vecs, n_clusters=6, pca_dim=50, random_state=0):

    if all_vecs.shape[0] == 0:
        return np.array([]), None, None

    X = all_vecs

    pca = None
    if pca_dim and X.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=random_state)
        Xr = pca.fit_transform(X)  # Xr has shape [num_patches, pca_dim]
    else:
        Xr = X

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)

    labels = km.fit_predict(Xr)  # labels shape: [num_patches]

    return labels, km, pca


def save_patch_crop(img_pil, mask, out_path):

    IMG = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE)))  # (H, W, 3), uint8
    m = Image.fromarray((mask * 255).astype(np.uint8)).resize(
        (IMG_SIZE, IMG_SIZE), resample=Image.NEAREST
    )
    m = np.array(m) > 0  # boolean mask at image scale

    ys, xs = np.where(m)
    if len(ys) == 0:
        return False

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    crop = IMG[y0:y1, x0:x1]

    if crop.size == 0:
        return False

    Image.fromarray(crop).save(out_path)
    return True

def build_ace_concepts(face_file_paths, n_segments=40, n_clusters=6, top_per_cluster=40, seed=0):

    random.seed(seed); np.random.seed(seed)

    vecs_all, meta = [], []   # meta holds tuples of (pil, mask) aligned with vecs_all
    for p in face_file_paths:
        pil = Image.open(p).convert("RGB")

        # 1) Over-segment the image into superpixels
        seg = segment_image(pil, n_segments=n_segments)

        # 2) Turn image into tensor and extract patch embeddings for all superpixels
        tens = transform(pil)
        vecs, masks = extract_patch_embeddings(pil, tens, seg)

        # Accumulate each patch vector and its mask
        for i in range(len(masks)):
            vecs_all.append(vecs[i])
            meta.append((pil, masks[i]))

    # If we had no patches at all, bail out early
    if len(vecs_all) == 0:
        print("No patches found. Check face images.")
        return [], np.array([])

    # Stack into a matrix [num_patches, C] for clustering
    vecs_all = np.stack(vecs_all)

    # 3) Cluster patch embeddings into n_clusters (PCA+KMeans inside)
    labels, km, pca = cluster_embeddings(vecs_all, n_clusters=n_clusters, pca_dim=50)

    # 4) For each cluster, save up to top_per_cluster patch crops into concepts/ace_k/
    ace_names = []
    for k in range(n_clusters):
        idx = np.where(labels == k)[0]  # indices of patches in this cluster
        if len(idx) < 10:
            # Skip very small clusters (too few examples to be a concept)
            continue

        sel = idx[:top_per_cluster]     # take the first N (could sort by centroid distance if desired)
        cname = f"ace_{k}"
        cdir = CONCEPTS_PATH / cname
        cdir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for j, ridx in enumerate(sel):
            pil, mask = meta[ridx]
            ok = save_patch_crop(pil, mask, cdir / f"patch_{j}.jpg")
            if ok:
                saved += 1

        # Only register this as a concept if we saved a decent number of patches
        if saved >= 10:
            ace_names.append(cname)
            print(f"{cname}: saved {saved} patches.")

    return ace_names, labels



_, face_files = load_image_tensors_multi_dir(
    data_dirs,
    transform_func=transform
)

print("Face files:", len(face_files))

# Limit to reasonable number for ACE (faster concept discovery)
max_ace_images = 500
face_files = face_files[:max_ace_images]
print(f"Using {len(face_files)} face files for ACE concept discovery")

# If nothing was loaded, give a clear hint and stop early.
if len(face_files) == 0:
    raise RuntimeError(
        f"No face images found under {FACE_DATA_PATHS}. "
        "Add .jpg/.jpeg/.png files to proceed."
    )


# Discover ACE concepts from the face images
ace_names, ace_labels = build_ace_concepts(
    face_files,
    n_segments=60,       # balanced segmentation
    n_clusters=5,        # reasonable number of concepts  
    top_per_cluster=50,  # good variety per concept
    seed=0
)
print("ACE concepts discovered:", ace_names)


def show_concept(cname, k=12):

    cdir = CONCEPTS_PATH / cname

    # Collect all image files under concepts/<cname> (recursively)
    imgs = [p for p in cdir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    # If the concept has no saved patches, let the user know and exit.
    if not imgs:
        print(f"{cname}: no images found.")
        return

    # Randomize order and cap to k images
    random.shuffle(imgs)
    imgs = imgs[:k]

    # Compute a compact grid size (≤6 columns)
    cols = min(k, 6)
    rows = (len(imgs) + cols - 1) // cols

    # Create the figure
    plt.figure(figsize=(2.5 * cols, 2.5 * rows))

    # Plot each image; skip any corrupted files gracefully
    for i, p in enumerate(imgs, 1):
        plt.subplot(rows, cols, i)
        try:
            plt.imshow(Image.open(p).convert("RGB"))
        except Exception as e:
            plt.imshow(np.zeros((10, 10, 3), dtype=np.uint8))  # tiny placeholder
            plt.title("read error", fontsize=8)
        plt.axis("off")

    plt.suptitle(f"{cname} — {len(imgs)} shown")
    plt.tight_layout()
    plt.savefig(f'./exp_output/tcav_and_ace_plots/ace_concept_{cname}.png', dpi=300, bbox_inches='tight')
    print(f"Saved ACE concept visualization to ./exp_output/tcav_and_ace_plots/ace_concept_{cname}.png")
    plt.close()

# Show the first few ACE concepts (if any exist)
for cname in ace_names[:3]:
    show_concept(cname, k=12)

def show_one_per_concept(concept_names):
    if not concept_names: # Add a check for empty list
        print("No concepts to display.")
        return
    cols = min(len(concept_names), 6)
    rows = (len(concept_names) + cols - 1) // cols
    plt.figure(figsize=(3.2*cols, 3.2*rows))
    for i, cname in enumerate(concept_names, 1):
        cdir = CONCEPTS_PATH / cname
        imgs = [p for p in cdir.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}]
        if not imgs: continue
        plt.subplot(rows, cols, i)
        plt.imshow(Image.open(imgs[0]).convert("RGB"))
        plt.title(cname)
        plt.axis("off")
    plt.suptitle("ACE concepts (first patch in each)")
    plt.savefig('./exp_output/tcav_and_ace_plots/ace_concepts_overview.png', dpi=300, bbox_inches='tight')
    print("Saved ACE concepts overview to ./exp_output/tcav_and_ace_plots/ace_concepts_overview.png")
    plt.close()

show_one_per_concept(ace_names)

# Create ACE concept objects for TCAV analysis
ace_concepts = [assemble_concept(name, i, concepts_path="./tcav_data/concept") for i, name in enumerate(ace_names)]

def show_concept_loader(concept, n=12):
    # Grab up to n tensors from the dataloader
    xs = []
    for batch in concept.data_iter:
        xs.append(batch)
        if sum(len(b) for b in xs) >= n:
            break
    if not xs:
        print(concept.id, "empty");
        return
    x = torch.cat(xs, dim=0)[:n]
    grid = make_grid(x, nrow=min(n, 6), normalize=True, pad_value=1.0)
    plt.figure(figsize=(10, 4))
    plt.imshow(np.transpose(grid.numpy(), (1,2,0)))
    plt.axis("off")
    plt.title(concept.id)
    plt.savefig(f'./exp_output/tcav_and_ace_plots/ace_concept_loader_{concept.id}.png', dpi=300, bbox_inches='tight')
    print(f"Saved ACE concept loader visualization to ./exp_output/tcav_and_ace_plots/ace_concept_loader_{concept.id}.png")
    plt.close()

for c in ace_concepts[:3]:
    show_concept_loader(c, n=12)


def plot_cluster_sizes(labels, title="ACE cluster sizes"):
    if labels is None or len(labels) == 0:
        print("No labels to plot.")
        return
    uniq, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    uniq, counts = uniq[order], counts[order]

    plt.figure(figsize=(8, 3))
    plt.bar([f"ace_{k}" for k in uniq], counts)
    plt.ylabel("# patches")
    plt.title(title)
    plt.xticks(rotation=30)
    plt.savefig('./exp_output/tcav_and_ace_plots/ace_cluster_sizes.png', dpi=300, bbox_inches='tight')
    print("Saved ACE cluster sizes plot to ./exp_output/tcav_and_ace_plots/ace_cluster_sizes.png")
    plt.close()


plot_cluster_sizes(ace_labels)

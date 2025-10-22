import os
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.segmentation import quickshift

import torch
import torch.nn.functional as F
import torchvision.models as models

from lime import lime_image
import shap

from utils import get_transforms, age_to_group

SEED = 42

n_correct = 5
n_incorrect = 5

lime_num_samples = 1000
lime_num_superpixels = 50
shap_background_samples = 20
shap_n_positive = 10

out_dir = Path("./exp_output/lime_and_shap_explanations")
model_ckpt = Path("./train_output/best_resnet18_gender.pth")
test_csv = Path("./train_output/test_predictions.csv")
device = "cuda" if torch.cuda.is_available() else "cpu"

transform_normalize = get_transforms(train=False)


def load_model(checkpoint_path, device=device):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, 2)
    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state.get("model_state", state) if isinstance(state, dict) else state
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def model_predict_numpy(images_np, model, device=device):
    model.eval()
    batch = []
    for img in images_np:
        pil = Image.fromarray(img.astype("uint8"), "RGB")
        t = transform_normalize(pil)
        batch.append(t)
    batch = torch.stack(batch).to(device)
    with torch.no_grad():
        out = model(batch)
        probs = F.softmax(out, dim=1).cpu().numpy()
    return probs


def plot_and_save_masked(img_pil, mask, filepath, alpha=0.7, cmap="jet"):
    arr = np.array(img_pil).astype(np.float32)/255.0
    mask_norm = (mask - mask.min()) / (mask.max() - mask.min() + 1e-9)
    cmap_obj = plt.get_cmap(cmap)
    heat = cmap_obj(mask_norm)[:,:,:3]  # H,W,3
    blended = arr * (1-alpha) + heat * alpha
    plt.figure(figsize=(4,4))
    plt.imshow(np.clip(blended, 0, 1))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight", pad_inches=0)
    plt.close()


def load_test_predictions(csv_path):
    def clean_int(x):
        if isinstance(x, (int, float)):
            return int(x)
        if isinstance(x, str):
            x = x.strip()
            if x.startswith("tensor(") and x.endswith(")"):
                x = x[len("tensor("):-1]
            x = x.replace("'", "").replace("\"", "")
            try:
                return int(float(x))
            except ValueError:
                return -1
        return -1
    rows = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            r["age"] = clean_int(r["age"])
            r["gender_true"] = clean_int(r["gender_true"])
            r["race"] = clean_int(r["race"])
            r["pred_gender"] = clean_int(r["pred_gender"])
            r["correct"] = (r["gender_true"] == r["pred_gender"])
            r["age_group"] = age_to_group(r["age"])
            rows.append(r)
    return rows


def select_examples(rows, age_group=None, race=None, correct=True, n=5):
    filtered = [r for r in rows if (age_group is None or r["age_group"] == age_group) and (race is None or r["race"] == race) and r["correct"] == correct]
    return filtered[:n]


# LIME explanations
def run_lime_on_image(pil_img, model, outpath, num_samples=lime_num_samples, num_superpixels=lime_num_superpixels):
    
    explainer = lime_image.LimeImageExplainer(random_state=SEED)
    img_arr = np.array(pil_img)

    def predict_fn(imgs):
        return model_predict_numpy(imgs, model)

    explanation = explainer.explain_instance(
        img_arr, 
        classifier_fn=predict_fn, 
        labels=(0,1), 
        top_labels=2,
        hide_color=0,
        num_samples=num_samples,
        segmentation_fn=lambda img: quickshift(img, kernel_size=3, max_dist=200, ratio=0.2))

    top_labels = explanation.top_labels
    for label in top_labels:
        temp, mask = explanation.get_image_and_mask(label, positive_only=True, num_features=10, hide_rest=False)
        outfn = outpath / f"lime_label{label}.png"
        plot_and_save_masked(Image.fromarray(temp), mask, outfn)

    return explanation


# SHAP explanations
def run_shap_on_images(pil_images, model, outdir, background_images=None):

    os.makedirs(outdir, exist_ok=True)
    images, filenames = pil_images
    bg_tensors = [transform_normalize(im).unsqueeze(0) for im in background_images]
    bg_batch = torch.cat(bg_tensors, dim=0).to(device)

    explainer = shap.GradientExplainer(model, bg_batch)

    results = []
    for i, im in enumerate(images):
        try:
            x = transform_normalize(im).unsqueeze(0).to(device)
            x.requires_grad_(True)
            shap_values = explainer.shap_values(x)

            sv_items = []
            for sv in shap_values:
                arr = np.array(sv)
                sv_items.append(arr)

            per_class_arrays = []
            for arr in sv_items:
                arr = np.asarray(arr)
                per_class_arrays.append(arr[0])  # arr shape == 1, C, H, W, per_class_arrays shape == C,H,W

            attributions = {}
            if per_class_arrays:
                img_array = np.array(im)
                img_normalized = img_array.astype(np.float32) / 255.0
                
                shap_values_for_plot = []
                for class_idx, c_arr in enumerate(per_class_arrays):
                    c_arr_hwc = np.transpose(c_arr, (1, 2, 0))  # C, H, W -> H, W, C
                    c_arr_batch = np.expand_dims(c_arr_hwc, axis=0)  # Add batch dimension: H, W, C -> 1, H, W, C
                    shap_values_for_plot.append(c_arr_batch)
                    
                    abs_arr = np.abs(c_arr)
                    map2d = np.mean(abs_arr, axis=0)
                    attributions[class_idx] = map2d
                
                img_batch = np.expand_dims(img_normalized, axis=0)  # Add batch dimension: H, W, C -> 1, H, W, C
                
                class_idx = 1
                shap_vals = shap_values_for_plot[class_idx]

                orig_filename = Path(filenames[i]).stem + ".png"
                outfn = outdir / orig_filename

                fig = plt.figure(figsize=(8, 4))
                shap.image_plot(shap_vals, img_batch, show=False)
                plt.savefig(outfn, bbox_inches="tight", pad_inches=0.1, dpi=150)
                plt.close(fig) 
                plt.clf()

            results.append(attributions)

        except Exception as e:
            print(f"SHAP failed for image index {i}: {e}")
            results.append({})

    return results


def aggregate_shap_maps(shap_maps_list, method="mean_abs"):
    arr = np.stack(shap_maps_list, axis=0)
    if method == "mean_abs":
        return np.mean(arr, axis=0)
    elif method == "median":
        return np.median(arr, axis=0)
    else:
        return np.mean(arr, axis=0)


def main():

    model = load_model(model_ckpt, device=device)
    rows = load_test_predictions(test_csv)

    subgroup_requests = [
        {"age_group": None, "race": None},
        {"age_group": "young_adult", "race": 3},
        {"age_group": "child", "race": 1},
    ]

    # out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    for sg in subgroup_requests:
        age_g = sg.get("age_group")
        race_g = sg.get("race")

        correct_samples = select_examples(rows, age_group=age_g, race=race_g, correct=True, n=n_correct)
        incorrect_samples = select_examples(rows, age_group=age_g, race=race_g, correct=False, n=n_incorrect)

        print(f"Subgroup age={age_g} race={race_g}: found {len(correct_samples)} correct, {len(incorrect_samples)} incorrect.")

        sg_name = f"age_{age_g or 'all'}__race_{race_g or 'all'}"
        lime_dir = out_dir / "lime" / sg_name
        shap_dir = out_dir / "shap" / sg_name
        os.makedirs(lime_dir, exist_ok=True)
        os.makedirs(shap_dir, exist_ok=True)

        for correctness_flag, label in [(True, "correct"), (False, "incorrect")]:
            sample_list = correct_samples if correctness_flag else incorrect_samples
            sub_lime_dir = lime_dir / label
            os.makedirs(sub_lime_dir, exist_ok=True)

            for r in sample_list:
                img_path = r["filepath"]
                pil = Image.open(img_path).convert("RGB")
                expl_dir = sub_lime_dir / Path(r["basename"]).stem
                os.makedirs(expl_dir, exist_ok=True)
                print(f"Running LIME on ({label})", r["basename"])
                run_lime_on_image(pil, model, expl_dir)


        # SHAP
        for correctness_flag, label in [(True, "correct"), (False, "incorrect")]:
            group_rows = [r for r in rows
                          if (r["correct"] == correctness_flag) and
                             (age_g is None or r["age_group"] == age_g) and
                             (race_g is None or r["race"] == race_g)]
            if len(group_rows) == 0:
                print(f"No {label} images for SHAP aggregation in this subgroup.")
                continue

            group_imgs = [Image.open(r["filepath"]).convert("RGB") for r in group_rows]
            group_filenames = [r["basename"] for r in group_rows]
            bg_images = group_imgs[:min(len(group_imgs), shap_background_samples)]
            sample_images = group_imgs[:min(len(group_imgs), shap_n_positive)]
            sample_filenames = group_filenames[:min(len(group_imgs), shap_n_positive)]
            print(f"Computing SHAP for {len(sample_images)} {label} images with {len(bg_images)} background samples.")

            shap_results = run_shap_on_images((sample_images, sample_filenames), model, shap_dir / label, background_images=bg_images)

            class_maps = []
            for atts in shap_results:
                if 1 in atts:
                    class_maps.append(atts[1])
                elif len(atts) > 0:
                    class_maps.append(np.mean(list(atts.values()), axis=0))

            if class_maps:
                agg_map = aggregate_shap_maps(class_maps, method="mean_abs")
                os.makedirs(shap_dir / label, exist_ok=True)
                plot_and_save_masked(sample_images[0], agg_map, shap_dir / label / "aggregated_shap_class1.png", alpha=0.6)
            else:
                print(f"No valid SHAP attributions found for {label} subgroup {sg}.")

    print("All explanations saved in", out_dir)

if __name__ == "__main__":
    main()

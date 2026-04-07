import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np
import pickle
from tqdm import tqdm
import glob
import matplotlib.tri as tri
from PIL import Image
import matplotlib.pyplot as plt

def fig2data(fig):
    """
    Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it.
    """
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)  # ARGB -> RGBA
    image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    image = np.asarray(image)
    return image


def render(args):
    i, result, crds, triang, v_max, v_min = args
    skip = 5
    step = i * skip
    target = result[1][step]
    predicted = result[0][step]

    fig, axes = plt.subplots(2, 1, figsize=(17, 8))
    target_v = np.linalg.norm(target, axis=-1)
    predicted_v = np.linalg.norm(predicted, axis=-1)

    for ax in axes:
        ax.cla()
        ax.triplot(triang, 'o-', color='k', ms=0.5, lw=0.3)

    handle1 = axes[0].tripcolor(triang, target_v, vmax=v_max, vmin=v_min)
    axes[1].tripcolor(triang, predicted_v, vmax=v_max, vmin=v_min)

    axes[0].set_title('Target\nTime @ %.2f s' % (step * 0.01))
    axes[1].set_title('Prediction\nTime @ %.2f s' % (step * 0.01))
    fig.colorbar(handle1, ax=[axes[0], axes[1]])

    img = fig2data(fig)[:, :, :3]  # Remove alpha channel
    plt.close(fig)
    return img, i


if __name__ == '__main__':
    result_files = glob.glob('result/*.pkl')
    os.makedirs('videos', exist_ok=True)

    for index, file in enumerate(result_files):
        with open(file, 'rb') as f:
            result, crds = pickle.load(f)
        triang = tri.Triangulation(crds[:, 0], crds[:, 1])

        file_name = 'videos/output%d.mp4' % index
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(file_name, fourcc, 20.0, (1700, 800))

        r_t = result[1]
        v_max = np.max(r_t)
        v_min = np.min(r_t)

        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(render, (i, result, crds, triang, v_max, v_min)): i for i in range(600 // 5)}
            frames = {}
            for future in tqdm(as_completed(futures), total=len(futures)):
                img, i = future.result()
                frames[i] = img
        # Write frames in correct order — as_completed() returns in completion
        # order which is non-deterministic; sorting by index fixes scrambled video
        for i in sorted(frames.keys()):
            img_resized = cv2.resize(frames[i], (1700, 800))
            out.write(img_resized)

        out.release()
        print('video %s saved' % file_name)
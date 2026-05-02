from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64

app = Flask(__name__)
CORS(app)

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)

def detect_wall_mask(img, click_x, click_y, tolerance=25):
    h, w = img.shape[:2]

    # ── 1. Strong edge map (furniture/objects have strong edges) ─────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 20, 80)
    
    # Thicken edges to form solid barriers flood fill cannot cross
    barrier_k = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    barriers   = cv2.dilate(edges, barrier_k, iterations=2)

    # ── 2. Flood fill from click, blocked by edge barriers ───────────────────
    fill_src = gray.copy()
    fill_src[barriers > 0] = 255   # make barriers bright — different from wall

    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    lo = (tolerance,); hi = (tolerance,)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    cv2.floodFill(fill_src, flood_mask,
                  (click_x, click_y), 255, lo, hi, flags)
    flood_region = flood_mask[1:-1, 1:-1]

    # ── 3. Horizon line heuristic: walls are typically ABOVE floor line ───────
    # Estimate floor line = bottom 25% of image is likely floor
    floor_line = int(h * 0.75)
    horizon_mask = np.zeros((h, w), np.uint8)
    horizon_mask[:floor_line, :] = 255   # only upper 75%

    # ── 4. Structural analysis: walls are large flat uniform regions ──────────
    # Compute local variance — walls have LOW variance (flat), objects HIGH
    img_gray_f = gray.astype(np.float32)
    local_mean  = cv2.blur(img_gray_f, (15, 15))
    local_mean2 = cv2.blur(img_gray_f ** 2, (15, 15))
    variance    = local_mean2 - local_mean ** 2
    variance    = np.clip(variance, 0, None)
    
    # Low variance = flat surface (likely wall)
    flat_mask = (variance < 300).astype(np.uint8) * 255

    # ── 5. LAB color similarity from click point ──────────────────────────────
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = img_lab[click_y, click_x]
    diff_lab = np.sqrt(np.sum((img_lab - target_lab) ** 2, axis=2))
    lab_mask = (diff_lab < tolerance * 2.0).astype(np.uint8) * 255

    # ── 6. Combine: flood fill AND (flat surface OR color similar) ────────────
    flat_or_color = cv2.bitwise_or(flat_mask, lab_mask)
    combined      = cv2.bitwise_and(flood_region, flat_or_color)

    # If too restrictive, fall back to flood only
    if cv2.countNonZero(combined) < cv2.countNonZero(flood_region) * 0.2:
        combined = flood_region

    # Apply horizon — remove floor area
    if click_y < floor_line:  # click is above floor, apply horizon filter
        combined = cv2.bitwise_and(combined, horizon_mask)

    # ── 7. Keep only the connected component at click point ──────────────────
    num_lbl, lbl_map = cv2.connectedComponents(combined, connectivity=8)
    click_lbl = lbl_map[click_y, click_x]

    wall_mask = np.zeros((h, w), np.uint8)
    if click_lbl > 0:
        wall_mask[lbl_map == click_lbl] = 255
    else:
        # fallback: largest component
        best = 0; best_area = 0
        for lid in range(1, num_lbl):
            area = np.sum(lbl_map == lid)
            if area > best_area:
                best_area = area; best = lid
        if best > 0:
            wall_mask[lbl_map == best] = 255

    # ── 8. Expand slightly to fill hairline gaps (picture frames etc.) ────────
    expand_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, expand_k)

    # Fill holes inside the wall region
    cnts, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled  = np.zeros_like(wall_mask)
    min_a   = h * w * 0.002
    for c in cnts:
        if cv2.contourArea(c) > min_a:
            cv2.drawContours(filled, [c], -1, 255, cv2.FILLED)
    if cv2.countNonZero(filled) > 0:
        wall_mask = filled

    # ── 9. Remove noise ───────────────────────────────────────────────────────
    open_k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, open_k)

    # ── 10. Erode slightly to avoid bleeding onto furniture edges ─────────────
    erode_k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    wall_mask = cv2.erode(wall_mask, erode_k, iterations=1)

    return wall_mask


def apply_wall_color(img, click_x, click_y, color_hex, tolerance=25, opacity=0.70):
    wall_mask = detect_wall_mask(img, click_x, click_y, tolerance)

    # Feather edges
    feathered = cv2.GaussianBlur(wall_mask.astype(np.float32), (11, 11), 0)

    bgr_color  = hex_to_bgr(color_hex)
    color_layer = np.full_like(img, bgr_color, dtype=np.float32)

    alpha   = (feathered / 255.0) * opacity
    alpha_3 = np.stack([alpha, alpha, alpha], axis=2)

    result = img.astype(np.float32) * (1 - alpha_3) + color_layer * alpha_3
    result = np.clip(result, 0, 255).astype(np.uint8)

    h, w   = img.shape[:2]
    coverage = round(cv2.countNonZero(wall_mask) / (h * w) * 100, 1)
    return result, coverage


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'PaintMyWalls backend running!'})


@app.route('/apply-color', methods=['POST'])
def apply_color():
    try:
        data    = request.get_json()
        arr     = np.frombuffer(base64.b64decode(data['image']), np.uint8)
        img     = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Invalid image'}), 400

        h, w = img.shape[:2]
        max_dim = 600
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img   = cv2.resize(img, (int(w * scale), int(h * scale)))
            h, w  = img.shape[:2]

        cx = max(2, min(int(float(data['x']) * w), w - 3))
        cy = max(2, min(int(float(data['y']) * h), h - 3))

        result, coverage = apply_wall_color(
            img, cx, cy,
            data.get('color', '#87a878'),
            int(data.get('tolerance', 25)),
            float(data.get('opacity', 0.70))
        )

        _, buf = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 93])
        return jsonify({
            'success':  True,
            'image':    base64.b64encode(buf).decode(),
            'width':    w, 'height': h,
            'coverage': coverage
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("🎨 PaintMyWalls Backend — Smart Wall Detection")
    print("✅ http://localhost:5000/health")
    app.run(debug=True, port=5000)
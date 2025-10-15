# file converter.py
import argparse, re
import numpy as np
import cv2
from skimage import measure
import pyclipper

# ---------- utils ----------
def is_near_white(rgb, thr=242):
    r,g,b = rgb; return (r>=thr and g>=thr and b>=thr)

def bgr_to_rgb_tuple(c): return (int(c[2]), int(c[1]), int(c[0]))

def auto_crop_foreground_lab(img_bgr,
                             border=8,          # bề rộng dải biên để ước lượng nền
                             deltaE_thr=None,   # ngưỡng ΔE; None = Otsu trên bản đồ ΔE
                             min_area_ratio=0.01,
                             margin=18):        # lề nới thêm quanh bbox
    """
    Cắt theo 'foreground ≠ background' (đo khoảng cách màu ΔE trong Lab đến nền).
    Ổn với nền xám/trắng/nhạt và cả ảnh ít bão hoà (tay người).
    """
    h, w = img_bgr.shape[:2]
    if min(h, w) < 2*border:  # ảnh quá nhỏ → bỏ cắt
        return img_bgr

    # 1) Ước lượng màu nền từ viền ảnh
    top    = img_bgr[:border, :]
    bottom = img_bgr[-border:, :]
    left   = img_bgr[:, :border]
    right  = img_bgr[:, -border:]
    border_pixels = np.vstack([
        top.reshape(-1,3), bottom.reshape(-1,3),
        left.reshape(-1,3), right.reshape(-1,3)
    ]).astype(np.uint8)

    # Lấy mode màu nền bằng kmeans nhỏ (K=2) rồi chọn cụm lớn hơn
    Z = border_pixels.astype(np.float32)
    K = 2
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels_b, centers_b = cv2.kmeans(Z, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    # cụm nền là cụm đông nhất
    bg_bgr = centers_b[np.bincount(labels_b.ravel()).argmax()].astype(np.uint8)[None,None,:]

    # 2) ΔE (Lab) so với nền
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_lab  = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)[0,0,:]  # (3,)
    dE = np.sqrt(np.sum((img_lab - bg_lab)**2, axis=2)).astype(np.float32)

    # 3) Ngưỡng ΔE → mask foreground
    if deltaE_thr is None:
        # Otsu trên ΔE (sau làm mượt nhẹ để ổn định)
        dE_blur = cv2.GaussianBlur(dE, (0,0), 1.0)
        dE_u8 = np.clip(dE_blur * (255.0 / max(dE_blur.max(), 1e-6)), 0, 255).astype(np.uint8)

        thr_val, _ = cv2.threshold(dE_u8, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # thr_val là ngưỡng Otsu trên [0,255], convert về thang ΔE thực
        deltaE_thr = (thr_val / 255.0) * float(dE_blur.max())

        # tránh ngưỡng quá thấp khi ảnh hơi nhiễu
        deltaE_thr = max(float(deltaE_thr), 6.0)



    mask = (dE > deltaE_thr).astype(np.uint8) * 255

    # 4) Làm sạch mask và lấy bbox
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k2)

    if cv2.countNonZero(mask) < min_area_ratio * h * w:
        # foreground quá nhỏ → không cắt
        return img_bgr

    ys, xs = np.where(mask > 0)
    y0, y1 = int(max(0, ys.min()-margin)), int(min(h, ys.max()+1+margin))
    x0, x1 = int(max(0, xs.min()-margin)), int(min(w, xs.max()+1+margin))

    return img_bgr[y0:y1, x0:x1]

def kmeans_colors(img_bgr, k=5, tile=384):
    h,w = img_bgr.shape[:2]
    small = cv2.resize(img_bgr, (max(1,w//2), max(1,h//2)), interpolation=cv2.INTER_AREA)
    Z = small.reshape((-1,3)).astype(np.float32)
    K = min(k, len(Z))
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _,_,centers = cv2.kmeans(Z, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)  # float32 BGR
    centers32 = centers.reshape(1,1,K,3).astype(np.float32)
    labels = np.empty((h,w), dtype=np.int32)
    for y0 in range(0,h,tile):
        y1 = min(h,y0+tile)
        for x0 in range(0,w,tile):
            x1 = min(w,x0+tile)
            blk = img_bgr[y0:y1, x0:x1].astype(np.float32)
            d2 = np.sum((blk[:,:,None,:]-centers32)**2, axis=3)
            labels[y0:y1, x0:x1] = np.argmin(d2, axis=2)
    colors_rgb = [bgr_to_rgb_tuple(c) for c in centers.astype(np.uint8)]
    return labels, colors_rgb

def is_mono_icon(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s_mean = float(hsv[...,1].mean())
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).ravel()
    total = hist.sum(); top2 = np.sort(hist)[-2:].sum()/max(total,1.0)
    return (s_mean < 25) and (top2 > 0.75)

# ---------- sub-pixel contour ----------
def find_subpixel_contours(mask_u8, sigma=0.8, level=0.5, min_len=20, close_eps=1.5):
    m = (mask_u8.astype("float32")/255.0)
    if sigma and sigma>0: m = cv2.GaussianBlur(m, (0,0), sigma)
    mp = np.pad(m, 1, mode="constant", constant_values=0.0)
    cs = measure.find_contours(mp, level=level)
    polys = []
    for c in cs:
        if c.shape[0] < min_len: continue
        xy = np.stack([c[:,1]-1.0, c[:,0]-1.0], axis=1).astype(np.float64)
        if np.linalg.norm(xy[0]-xy[-1]) <= close_eps: xy[-1] = xy[0].copy()
        polys.append(xy)
    return polys

# ---------- polygon repair / union ----------
def repair_union_polys(polys, clean_dist=0.35, scale=100.0):
    if not polys: return []
    sc = float(scale)
    subject = []
    for p in polys:
        if len(p) < 3: continue
        q = p.copy()
        if np.linalg.norm(q[0]-q[-1]) > 1e-6:
            q = np.vstack([q, q[0]])
        subject.append([(int(round(x*sc)), int(round(y*sc))) for x,y in q])
    if not subject: return []
    subject = [pyclipper.CleanPolygon(path, clean_dist*sc) for path in subject]
    subject = [s for s in subject if len(s)>=3]
    if not subject: return []
    pc = pyclipper.Pyclipper()
    pc.AddPaths(subject, pyclipper.PT_SUBJECT, True)
    tree = pc.Execute2(pyclipper.CT_UNION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
    out_paths, stack = [], [tree]
    while stack:
        n = stack.pop()
        if isinstance(n, list): stack += n
        else:
            if n.Contour: out_paths.append(n.Contour)
            if n.Childs: stack += n.Childs
    res = []
    for path in out_paths:
        res.append(np.array([(x/sc, y/sc) for x,y in path], dtype=np.float64))
    return res

# ---------- line-preserving + Bézier ----------
def rdp(points, eps):
    P = points.astype(np.float64)
    if len(P) < 3 or eps <= 0: return P
    def _rdp(seg):
        if len(seg) < 3: return seg
        a, b = seg[0], seg[-1]
        ab = b - a; ab2 = np.dot(ab, ab) + 1e-12
        d = np.abs(np.cross(ab, seg[1:-1] - a)) / np.sqrt(ab2)
        i = np.argmax(d); dmax = d[i]
        if dmax > eps:
            i += 1
            L = _rdp(seg[:i+1]); R = _rdp(seg[i:])
            return np.vstack([L[:-1], R])
        return np.vstack([a, b])
    return _rdp(P)

def fit_bezier_segment(pts, max_err=0.28):
    def _norm(v): n=np.linalg.norm(v); return v/n if n>1e-12 else v
    d = pts.astype(np.float64); n=len(d)
    if n<2: return []
    t1 = _norm(d[1]-d[0]); t2 = _norm(d[-2]-d[-1])
    def _chord(d):
        u=np.zeros(n); 
        for i in range(1,n): u[i]=u[i-1]+np.linalg.norm(d[i]-d[i-1])
        return u/u[-1] if u[-1]>0 else np.linspace(0,1,n)
    def _eval(p0,p1,p2,p3,t):
        mt=1-t; return (mt**3)*p0+3*mt*mt*t*p1+3*mt*t*t*p2+t**3*p3
    def _gen(u,th1,th2):
        p0,p3=d[0],d[-1]; C=np.zeros((2,2)); X=np.zeros(2)
        for j,t in enumerate(u):
            b1=3*(1-t)*(1-t)*t; b2=3*(1-t)*t*t
            a1=b1*th1; a2=b2*th2
            q0=(1-t)**3*p0+t**3*p3+b1*p0+b2*p3
            r=d[j]-q0
            C[0,0]+=a1@a1; C[0,1]+=a1@a2; C[1,1]+=a2@a2
            X[0]+=a1@r;   X[1]+=a2@r
        C[1,0]=C[0,1]; det=C[0,0]*C[1,1]-C[0,1]*C[1,0]
        if abs(det)<1e-12:
            L=np.linalg.norm(p3-p0)/3.0; return np.array([p0,p0+th1*L,p3-th2*L,p3])
        a=(X[0]*C[1,1]-X[1]*C[0,1])/det; b=(C[0,0]*X[1]-C[0,1]*X[0])/det
        if a<=1e-6 or b<=1e-6:
            L=np.linalg.norm(p3-p0)/3.0; return np.array([p0,p0+th1*L,p3-th2*L,p3])
        return np.array([p0,p0+th1*a,p3-th2*b,p3])
    def _maxerr(B,u):
        mx=0; sp=n//2
        for i in range(1,n-1):
            e=np.linalg.norm(_eval(*B,u[i])-d[i])
            if e>mx: mx=e; sp=i
        return mx, sp
    u=_chord(d); B=_gen(u,t1,t2); mx,sp=_maxerr(B,u)
    if mx<=max_err: return [B]
    for _ in range(12):
        p0,p1,p2,p3=B
        for j in range(n):
            t=u[j]; mt=1-t
            Q=(mt**3)*p0+3*mt*mt*t*p1+3*mt*t*t*p2+t**3*p3
            Q1=3*mt*mt*(p1-p0)+6*mt*t*(p2-p1)+3*t*t*(p3-p2)
            Q2=6*(1-t)*(p2-2*p1+p0)+6*t*(p3-2*p2+p1)
            num=(Q-d[j])@Q1; den=(Q1@Q1)+(Q-d[j])@Q2
            if abs(den)>1e-12: t-=num/den
            u[j]=min(max(t,0.0),1.0)
        B=_gen(u,t1,t2); mx,sp=_maxerr(B,u)
        if mx<=max_err: return [B]
    left = fit_bezier_segment(d[:sp+1], max_err)
    right= fit_bezier_segment(d[sp:],   max_err)
    return left+right

def build_path_preserve_lines(poly, corner_eps=0.9, line_tol=0.10, max_err=0.28):
    P = poly.astype(np.float64)
    # tìm điểm “góc” bằng RDP rồi map về index gần nhất (không cần scipy)
    K = rdp(P, eps=corner_eps)
    idxs = [0]
    for k in K[1:-1]:
        j = int(np.argmin(np.sum((P - k)**2, axis=1)))
        idxs.append(j)
    idxs.append(len(P)-1)
    idxs = sorted(set(idxs))

    segs = []
    for s,e in zip(idxs[:-1], idxs[1:]):
        seg = P[s:e+1]
        if len(seg) < 2: continue
        a,b = seg[0], seg[-1]
        L = np.linalg.norm(b-a)+1e-9
        dev = np.max(np.abs(np.cross(b-a, seg-a))/L)  # max lệch khỏi đường thẳng
        if (dev/L) <= line_tol:
            segs.append(("L", a, b))
        else:
            B = fit_bezier_segment(seg, max_err=max_err)
            for (P0,C1,C2,P3) in B:
                segs.append(("C", P0, C1, C2, P3))
    close = np.linalg.norm(P[0]-P[-1]) < 1e-6
    if not segs: return ""
    d = [f"M {segs[0][1][0]:.3f} {segs[0][1][1]:.3f}"]
    for s in segs:
        if s[0]=="L":
            _, a, b = s
            d.append(f"L {b[0]:.3f} {b[1]:.3f}")
        else:
            _, P0,C1,C2,P3 = s
            d.append(f"C {C1[0]:.3f} {C1[1]:.3f} {C2[0]:.3f} {C2[1]:.3f} {P3[0]:.3f} {P3[1]:.3f}")
    if close: d.append("Z")
    return " ".join(d)

def gaussian1d(arr, ksize=9, sigma=1.2):
    ksize = max(3, int(ksize) | 1)
    x = np.arange(ksize) - ksize//2
    g = np.exp(-(x*x)/(2*sigma*sigma)); g /= g.sum()
    pad = ksize//2
    a = np.pad(arr, (pad, pad), mode="wrap")  # wrap để giữ tính chu kỳ của contour đóng
    return np.convolve(a, g, mode="same")[pad:-pad]

def smooth_curve_only(pts, line_tol=0.10, sigma=1.2):
    """Làm mượt nhẹ CHỈ các đoạn cong, giữ nguyên đoạn thẳng."""
    P = pts.astype(np.float64)
    if len(P) < 6 or sigma <= 0:
        return P
    # độ lệch cục bộ so với dây cung
    a, b = P[:-1], P[1:]
    L = np.linalg.norm(b - a, axis=1) + 1e-9
    # dùng vector trung bình làm hướng tham chiếu đơn giản
    ref = (P.max(axis=0) + P.min(axis=0)) / 2.0
    dev = np.abs(np.cross(b - a, a - ref)) / L
    dev = np.r_[dev, dev[-1]]
    curved = (dev / (L.max() + 1e-9)) > line_tol * 0.7

    Q = P.copy()
    if curved.any():
        k = max(5, int(6 * sigma) | 1)
        for d in (0, 1):
            arr = P[:, d]
            sm = gaussian1d(arr, ksize=k, sigma=sigma)
            Q[curved, d] = sm[curved]
    return Q

# ---------- SVG ----------
def write_svg(paths, width, height, outfile, background=None):
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n')
        if background:
            f.write(f'  <rect x="0" y="0" width="{width}" height="{height}" fill="{background}"/>\n')
        for (fill_rgb, d) in paths:
            if not d: continue
            f.write(f'  <path d="{d}" fill="rgb({fill_rgb[0]},{fill_rgb[1]},{fill_rgb[2]})" fill-rule="evenodd"/>\n')
        f.write('</svg>\n')

# ---------- pipelines ----------
def pipeline_color(img_bgr, output_path, k_colors=4, min_area_px=80, area_max_ratio=0.92,
                       super_sample=3, decimate=2, subpixel_sigma=0.9,
                       fit_gauss_kernel=9, repair=True, repair_clean=0.35):
    h0,w0 = img_bgr.shape[:2]
    if super_sample>1:
        img_bgr = cv2.resize(img_bgr, (w0*super_sample, h0*super_sample), interpolation=cv2.INTER_CUBIC)
    img_bgr = cv2.bilateralFilter(img_bgr, 5, 50, 50)
    h,w = img_bgr.shape[:2]
    labels, colors_rgb = kmeans_colors(img_bgr, k=k_colors, tile=384)
    kept = []
    for idx, col in enumerate(colors_rgb):
        if is_near_white(col, 245): continue
        mask = (labels==idx).astype(np.uint8)*255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        num, comp, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        keepm = np.zeros_like(mask)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area_px: keepm[comp==i] = 255
        mask = keepm
        area = int(cv2.countNonZero(mask))
        if area == 0 or area >= area_max_ratio*w*h: continue
        polys = find_subpixel_contours(mask, sigma=subpixel_sigma, level=0.5, min_len=20)
        if not polys: continue
        if repair:
            polys = repair_union_polys(polys, clean_dist=repair_clean, scale=100.0)
            if not polys: continue
        d_parts = []
        for pts in polys:
            pts_f = pts.astype(np.float32)
            klen = fit_gauss_kernel if len(pts_f)>=fit_gauss_kernel else (len(pts_f)|1)
            if klen>=5:
                pts_f[:,0] = cv2.GaussianBlur(pts_f[:,0], (klen,1), 0.8).ravel()
                pts_f[:,1] = cv2.GaussianBlur(pts_f[:,1], (klen,1), 0.8).ravel()
            if decimate>1:
                pts_f = pts_f[::decimate]
                if len(pts_f)<8: continue
            closed = np.linalg.norm(pts_f[0]-pts_f[-1]) < 1e-6
            d = build_path_preserve_lines(pts_f, corner_eps=0.9, line_tol=0.10, max_err=0.28) if closed else \
                build_path_preserve_lines(pts_f, corner_eps=0.9, line_tol=0.10, max_err=0.28)
            if d: d_parts.append(d)
        if not d_parts: continue
        kept.append((col, " ".join(d_parts), area))
    if not kept: raise SystemExit("Không có path hợp lệ (color).")
    if super_sample>1:
        def rescale(d,s): return re.sub(r"-?\d+(?:\.\d+)?", lambda m: f"{float(m.group())/s:.3f}", d)
        kept = [(col, rescale(d, super_sample), area/(super_sample**2)) for col,d,area in kept]
        w,h = w0,h0
    kept.sort(key=lambda x: x[2])
    write_svg([(c,d) for c,d,_ in kept], w, h, output_path, background=None)

def pipeline_mono(img_bgr, output_path, super_sample=3, min_area_px=40,
                  subpixel_sigma=0.8, decimate=1, repair=True, repair_clean=0.3,
                  morph_close=3, morph_open=0, invert_auto=True,
                  corner_eps=0.9, line_tol=0.10, max_err=0.28,
                  curve_sigma=1.2):   # <— thêm tham số này
    h0, w0 = img_bgr.shape[:2]
    if super_sample > 1:
        img_bgr = cv2.resize(img_bgr, (w0*super_sample, h0*super_sample), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if invert_auto and cv2.countNonZero(mask) > mask.size // 2:
        mask = 255 - mask

    if morph_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    if morph_open > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    num, comp, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keepm = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
            keepm[comp == i] = 255
    mask = keepm

    h, w = img_bgr.shape[:2]
    polys = find_subpixel_contours(mask, sigma=subpixel_sigma, level=0.5, min_len=20)
    if not polys:
        raise SystemExit("Không có biên (mono).")
    if repair:
        polys = repair_union_polys(polys, clean_dist=repair_clean, scale=100.0)
        if not polys:
            raise SystemExit("PyClipper trả rỗng (mono).")

    d_parts = []
    for pts in polys:
        if decimate > 1:
            pts = pts[::decimate]
            if len(pts) < 8:
                continue

        # mài răng CHỈ ở cung cong (giữ đoạn thẳng):
        if curve_sigma and curve_sigma > 0:
            pts = smooth_curve_only(pts, line_tol=line_tol, sigma=curve_sigma)

        d = build_path_preserve_lines(
            pts, corner_eps=corner_eps, line_tol=line_tol, max_err=max_err
        )
        if d:
            d_parts.append(d)

    if not d_parts:
        raise SystemExit("Không tạo được path (mono).")

    d_all = " ".join(d_parts)

    # màu fill: trung bình vùng foreground
    fg = img_bgr[mask > 0]
    mean_bgr = fg.mean(axis=0) if fg.size else np.array([0, 0, 0], dtype=np.float32)
    fill_rgb = bgr_to_rgb_tuple(mean_bgr)

    if super_sample > 1:
        d_all = re.sub(r"-?\d+(?:\.\d+)?", lambda m: f"{float(m.group())/super_sample:.3f}", d_all)
        w, h = w0, h0

    write_svg([(fill_rgb, d_all)], w, h, output_path, background=None)

# ---------- master ----------
def convert_png_to_svg(input_path, output_path,
                             img_bgr_in=None,
                             mode="auto",
                             k_colors=4,
                             min_area_px=80,
                             area_max_ratio=0.92,
                             super_sample=3,
                             decimate=1,
                             auto_crop=True,
                             manual_crop=None,
                             subpixel_sigma=0.8,
                             repair=True,
                             repair_clean=0.3,
                             corner_eps=0.9,
                             line_tol=0.10,
                             max_err=0.28,
                             morph_close=4,
                             morph_open=1,
                             curve_sigma=1.4):
    if img_bgr_in is not None:
        img_bgr = img_bgr_in
    else:
        img_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)

    if img_bgr is None: raise SystemExit("Không đọc được ảnh đầu vào.")
    if manual_crop:
        x,y,w,h = [int(v) for v in manual_crop.split(",")]
        img_bgr = img_bgr[y:y+h, x:x+w]
    elif auto_crop:
        img_bgr = auto_crop_foreground_lab(img_bgr, border=8, deltaE_thr=None, min_area_ratio=0.01, margin=18)

    chosen = "mono" if (mode=="auto" and is_mono_icon(img_bgr)) else ("color" if mode=="color" else ("mono" if mode=="mono" else "color"))
    
    if chosen == "mono":
        pipeline_mono(img_bgr, output_path,
                      super_sample=super_sample, min_area_px=max(20, min_area_px//2),
                      subpixel_sigma=subpixel_sigma, decimate=decimate,
                      repair=repair, repair_clean=repair_clean,
                      morph_close=morph_close,
                      morph_open=morph_open,
                      corner_eps=corner_eps, line_tol=line_tol, max_err=max_err,
                      curve_sigma=curve_sigma)
    else:
        pipeline_color(img_bgr, output_path,
                       k_colors=k_colors, min_area_px=min_area_px, area_max_ratio=area_max_ratio,
                       super_sample=super_sample, decimate=max(1,decimate),
                       subpixel_sigma=max(0.8, subpixel_sigma),
                       fit_gauss_kernel=9, repair=repair, repair_clean=repair_clean)

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PNG/JPEG -> SVG (auto color/mono), biên mượt & giữ cạnh thẳng.")
    ap.add_argument("--input", type=str, default="image2.png", help="Input PNG/JPEG image path.")
    ap.add_argument("--output", type=str, default="image2.svg", help="Output SVG file path.")
    ap.add_argument("--mode", choices=["auto","color","mono"], default="auto", help="Chế độ chuyển đổi: auto=color/mono, color=to color, mono=to mono.")
    ap.add_argument("--k", type=int, default=4, help="Số lượng màu sắc cho chế độ màu.")
    ap.add_argument("--min-area", type=int, default=80, help="Diện tích tối thiểu cho vùng foreground.")
    ap.add_argument("--area-max-ratio", type=float, default=0.92, help="Tỷ lệ tối đa của diện tích vùng foreground.")
    ap.add_argument("--super-sample", type=int, default=4, help="Hệ số siêu mẫu.")
    ap.add_argument("--decimate", type=int, default=1, help="Hệ số giảm mẫu.")
    ap.add_argument("--no-auto-crop", action="store_true", help="Tắt tự động cắt.")
    ap.add_argument("--crop", type=str, default=None, help="Cắt thủ công: x,y,w,h")
    ap.add_argument("--subpixel-sigma", type=float, default=0.8, help="Sigma cho làm mịn subpixel.")
    ap.add_argument("--no-repair", action="store_true", help="Tắt sửa chữa biên.")
    ap.add_argument("--repair-clean", type=float, default=0.01, help="Khoảng cách làm sạch cho sửa chữa biên.")
    ap.add_argument("--corner-eps", type=float, default=0.9, help="Sai số cho các góc.")
    ap.add_argument("--line-tol", type=float, default=0.1, help="Sai số cho các đường thẳng.")
    ap.add_argument("--max-err", type=float, default=0.26, help="Sai số tối đa cho các đường thẳng.")
    ap.add_argument("--morph-close", type=int, default=4, help="Kích thước kernel đóng hình thái học.")
    ap.add_argument("--morph-open", type=int, default=1, help="Kích thước kernel mở hình thái học.")
    ap.add_argument("--curve-sigma", type=float, default=1.4, help="Sigma Gaussian để mài răng CHỈ trên đoạn cong (mono). 0=tắt")
    args = ap.parse_args()

    convert_png_to_svg(
        args.input, args.output,
        mode=args.mode,
        k_colors=args.k,
        min_area_px=args.min_area,
        area_max_ratio=args.area_max_ratio,
        super_sample=args.super_sample,
        decimate=args.decimate,
        auto_crop=(not args.no_auto_crop),
        manual_crop=args.crop,
        subpixel_sigma=args.subpixel_sigma,
        repair=(not args.no_repair),
        repair_clean=args.repair_clean,
        corner_eps=args.corner_eps,
        line_tol=args.line_tol,
        max_err=args.max_err,
        morph_close=args.morph_close,
        morph_open=args.morph_open,
        curve_sigma=args.curve_sigma
    )
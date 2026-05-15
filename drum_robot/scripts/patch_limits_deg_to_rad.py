# patch_limits_deg_to_rad.py
# 목적:
# - USD 내 physics:lowerLimit / physics:upperLimit 이 "도(deg) 값"처럼 보이는 경우에만
#   라디안(rad)으로 일괄 변환해 새 USD로 저장한다.
# - (옵션) 지정한 joint들의 physics:jointState:angular:position(q0)도 세팅한다.
#
# 실행 예시:
# ./isaaclab.sh -p source/extensions/drum_robot/drum_robot/scripts/patch_limits_deg_to_rad.py \
#   --in  /.../drum_robot.usd \
#   --out /.../drum_robot_patched.usd
#
# q0까지 같이:
# ./isaaclab.sh -p .../patch_limits_deg_to_rad.py --in ... --out ... --set-q0

from isaaclab.app import AppLauncher

# Isaac Sim 컨텍스트 부팅(그래야 pxr 사용 가능)
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

print("[1] AppLauncher OK", flush=True)


def main():
    import argparse
    import math
    from pxr import Usd, Sdf

    def deg2rad(x: float) -> float:
        return x * math.pi / 180.0

    def getf(prim, name: str):
        a = prim.GetAttribute(name)
        if not a or not a.HasAuthoredValueOpinion():
            return None
        try:
            return float(a.Get())
        except Exception:
            return None

    def setf(prim, name: str, value: float):
        a = prim.GetAttribute(name)
        if not a:
            a = prim.CreateAttribute(name, Sdf.ValueTypeNames.Float)
        a.Set(float(value))

    def looks_like_deg(lo: float, hi: float) -> bool:
        """
        revolute joint limit이 rad면 보통 [-6.3, 6.3] 근처.
        30, 180, -90 같은 값이면 deg일 가능성이 매우 높음.

        휴리스틱:
        - |lo| or |hi| > 10 이면 deg로 간주
        - 또는 (hi - lo) > 10 이면 deg로 간주
        """
        if lo is None or hi is None:
            return False
        if abs(lo) > 10.0 or abs(hi) > 10.0:
            return True
        if (hi - lo) > 10.0:
            return True
        return False

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_usd", required=True)
    ap.add_argument("--out", dest="out_usd", required=True)
    ap.add_argument("--set-q0", action="store_true", help="옵션: q0도 같이 세팅")
    args = ap.parse_args()

    stage = Usd.Stage.Open(args.in_usd)
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {args.in_usd}")

    # ----------------------------
    # 1) deg->rad limit 일괄 변환
    # ----------------------------
    patched = []
    skipped = 0

    for prim in stage.Traverse():
        lo = getf(prim, "physics:lowerLimit")
        hi = getf(prim, "physics:upperLimit")
        if lo is None or hi is None:
            continue

        if looks_like_deg(lo, hi):
            new_lo = deg2rad(lo)
            new_hi = deg2rad(hi)
            setf(prim, "physics:lowerLimit", new_lo)
            setf(prim, "physics:upperLimit", new_hi)
            patched.append((str(prim.GetPath()), lo, hi, new_lo, new_hi))
        else:
            skipped += 1

    print("\n===== LIMIT PATCH SUMMARY =====", flush=True)
    print(f"patched joints: {len(patched)}", flush=True)
    print(f"skipped (already rad-like): {skipped}", flush=True)

    # 너무 많이 출력하면 지저분하니 상위 몇 개만
    for i, (p, lo, hi, nlo, nhi) in enumerate(patched[:20]):
        print(f"[{i:02d}] {p}  [{lo},{hi}] -> [{nlo:.6f},{nhi:.6f}] (deg->rad)", flush=True)
    if len(patched) > 20:
        print(f"... ({len(patched)-20} more patched)", flush=True)

    # ----------------------------
    # 2) (옵션) q0 세팅
    # ----------------------------
    if args.set_q0:
        # 네가 원한 초기 자세(도) -> rad로 저장
        init_deg = {
            "left_shoulder_1": 90,
            "left_shoulder_2": 0,
            "left_elbow": 90,
            "left_wrist": 90,
            "right_shoulder_1": 90,
            "right_shoulder_2": 0,
            "right_elbow": 90,
            "right_wrist": -90,
        }

        # name -> prim 찾기
        name_to_prim = {}
        for prim in stage.Traverse():
            name_to_prim.setdefault(prim.GetName(), []).append(prim)

        def pick(name: str):
            cands = name_to_prim.get(name, [])
            if not cands:
                return None
            for p in cands:
                if p.GetAttribute("physics:lowerLimit") or p.GetAttribute("physics:upperLimit"):
                    return p
            return cands[0]

        print("\n===== Q0 PATCH =====", flush=True)
        for jn, d in init_deg.items():
            prim = pick(jn)
            if prim is None:
                print(f"[Q0][WARN] prim not found: {jn}", flush=True)
                continue

            lo = getf(prim, "physics:lowerLimit")
            hi = getf(prim, "physics:upperLimit")
            q0 = deg2rad(float(d))

            # limit clamp (limit은 이미 rad로 변환된 상태여야 정상)
            if lo is not None and q0 < lo:
                q0 = lo + 1e-3
            if hi is not None and q0 > hi:
                q0 = hi - 1e-3

            setf(prim, "physics:jointState:angular:position", q0)
            print(f"[Q0] {jn} prim={prim.GetPath()} q0={q0:.6f} rad limit=[{lo},{hi}]", flush=True)

    # ----------------------------
    # 3) 새 USD로 저장
    # ----------------------------
    stage.GetRootLayer().Export(args.out_usd)
    print("\n===== DONE =====", flush=True)
    print("output:", args.out_usd, flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

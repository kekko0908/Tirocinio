from common import (
    EnvConfig,
    make_controller,
    load_json,
    print_ok,
    print_warn,
    get_rgb_bgr,
    data_dir,
)
import math
import cv2


def dist3(a, b):
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2)


def draw_hud(img_bgr, lines):
    y = 28
    for line in lines:
        cv2.putText(img_bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28
    return img_bgr


def save_step_frame(controller, out_dir, name, hud_lines):
    frame = get_rgb_bgr(controller.last_event)
    frame = draw_hud(frame, hud_lines)
    path = out_dir / f"{name}.png"
    cv2.imwrite(str(path), frame)


def get_arm(controller):
    return controller.last_event.metadata.get("arm", {})


def get_hand(controller):
    arm = get_arm(controller)
    return arm.get("handSphereCenter", None), arm.get("handSphereRadius", None)


def is_target_in_range(controller, target_id):
    arm = get_arm(controller)
    pickupable = arm.get("pickupableObjects", [])
    touched = arm.get("touchedNotHeldObjects", [])
    held = arm.get("heldObjects", [])
    in_pickupable = any(x.get("objectId") == target_id for x in pickupable)
    in_touched = any(x.get("objectId") == target_id for x in touched)
    return in_pickupable, in_touched, len(pickupable), len(touched), len(held)


def move_arm_world(controller, goal, speed=1):
    ev = controller.step(action="MoveArm", position=goal, coordinateSpace="world", speed=speed)
    ok = ev.metadata.get("lastActionSuccess", False)
    err = ev.metadata.get("errorMessage", "")
    return ok, err


def norm2(x, z):
    n = math.sqrt(x * x + z * z)
    if n < 1e-8:
        return 0.0, 0.0
    return x / n, z / n


def main():
    tele = load_json("arm_best_pose_state.json")

    scene = tele["scene"]
    target_id = tele["target"]["objectId"]
    target_type = tele["target"]["objectType"]

    agent_pos = tele["agent"]["position"]
    agent_rot = tele["agent"]["rotation"]
    horizon = tele["agent"]["horizon"]

    c = tele["bbox_center_world"]
    center = {"x": float(c["x"]), "y": float(c["y"]), "z": float(c["z"])}

    out_dir = data_dir("frames") / "10_arm"
    out_dir.mkdir(parents=True, exist_ok=True)

    controller = make_controller(EnvConfig(scene=scene, agent_mode="arm"))
    controller.step(action="Teleport", position=agent_pos, rotation=agent_rot, horizon=horizon)

    frame_idx = 0
    yaw = controller.last_event.metadata["agent"]["rotation"]["y"]

    hand0, r0 = get_hand(controller)
    d0 = dist3(hand0, center) if hand0 else None
    save_step_frame(
        controller,
        out_dir,
        f"{frame_idx:03d}_INIT",
        [
            f"target={target_type}",
            f"center=({center['x']:.2f},{center['y']:.2f},{center['z']:.2f})",
            f"hand_dist={d0:.3f}m r={r0}" if d0 is not None else "hand_dist=N/A",
            f"yaw={yaw:.1f}",
            "phase=INIT",
        ],
    )
    frame_idx += 1

    if hand0 is None:
        print_warn("handSphereCenter assente.")
        controller.stop()
        return

    # =========================================================
    # A) ESCI DAL BANCO: BACKOFF + SHIFT->AGENT
    # =========================================================
    rad = math.radians(yaw)
    fwd = (math.sin(rad), math.cos(rad))
    back = (-fwd[0], -fwd[1])

    backoff_list = [0.10, 0.20, 0.30, 0.40]
    shift_list = [0.10, 0.20, 0.30, 0.40]

    # RAISE PIÙ BASSO (come chiedi tu)
    hover_y = center["y"] + 0.35  # <<< ABBASSATO

    positioned = False
    for ti in range(len(backoff_list)):
        b = backoff_list[ti]
        s = shift_list[ti]

        hand, r = get_hand(controller)

        goal_back = {"x": hand["x"] + back[0] * b, "y": hand["y"], "z": hand["z"] + back[1] * b}
        ok, err = move_arm_world(controller, goal_back, speed=1)
        save_step_frame(controller, out_dir, f"{frame_idx:03d}_BACKOFF_{ti+1:02d}", [f"ok={ok} err={err[:80]}"])
        frame_idx += 1
        if not ok:
            continue

        hand, r = get_hand(controller)
        vx = agent_pos["x"] - hand["x"]
        vz = agent_pos["z"] - hand["z"]
        nx, nz = norm2(vx, vz)

        goal_shift = {"x": hand["x"] + nx * s, "y": hand["y"], "z": hand["z"] + nz * s}
        ok2, err2 = move_arm_world(controller, goal_shift, speed=1)
        save_step_frame(controller, out_dir, f"{frame_idx:03d}_SHIFT_{ti+1:02d}", [f"ok={ok2} err={err2[:80]}"])
        frame_idx += 1
        if not ok2:
            continue

        # mini-raise basso
        hand, r = get_hand(controller)
        goal_raise = {"x": hand["x"], "y": hover_y, "z": hand["z"]}
        ok3, err3 = move_arm_world(controller, goal_raise, speed=1)
        save_step_frame(controller, out_dir, f"{frame_idx:03d}_RAISE_LOW_{ti+1:02d}", [f"ok={ok3} err={err3[:80]}"])
        frame_idx += 1

        if ok3:
            positioned = True
            break

    if not positioned:
        print_warn("Non sono riuscito a posizionare braccio (backoff/shift/raise_low).")
        print_warn(f"Frames: {out_dir}")
        controller.stop()
        return

    # =========================================================
    # B) ALIGN_HOVER con TEST DISCESA
    # scegliamo solo goal che permettono almeno 3 step verso il basso senza collisione
    # =========================================================
    scan = [-0.14, -0.10, -0.06, -0.03, 0.00, 0.03, 0.06, 0.10, 0.14]
    test_down = [0.28, 0.22, 0.16]  # 3 step sotto hover (relative al center.y)

    best_goal = None
    best_score = 1e9  # score = hand_dist + penalty

    for dx in scan:
        for dz in scan:
            goal_hover = {"x": center["x"] + dx, "y": hover_y, "z": center["z"] + dz}
            okh, erh = move_arm_world(controller, goal_hover, speed=1)

            hand, r = get_hand(controller)
            d = dist3(hand, center) if hand else 999.0

            # test discesa: proviamo 3 quote più basse, ma torniamo subito su hover
            down_ok = 0
            last_down_err = ""
            if okh:
                for yo in test_down:
                    gtest = {"x": goal_hover["x"], "y": center["y"] + yo, "z": goal_hover["z"]}
                    okd, erd = move_arm_world(controller, gtest, speed=1)
                    if okd:
                        down_ok += 1
                    else:
                        last_down_err = erd
                        break
                # ritorno su hover candidate
                move_arm_world(controller, goal_hover, speed=1)

            score = d + (0 if down_ok == len(test_down) else 999.0)

            save_step_frame(
                controller,
                out_dir,
                f"{frame_idx:03d}_HOVER_dx{dx:+.2f}_dz{dz:+.2f}",
                [
                    "phase=ALIGN_HOVER+DOWNTEST",
                    f"goal=({goal_hover['x']:.2f},{goal_hover['y']:.2f},{goal_hover['z']:.2f})",
                    f"hover_ok={okh} " + (f"err={erh[:55]}" if (not okh and erh) else ""),
                    f"hand_dist={d:.3f} r={r}",
                    f"down_ok={down_ok}/{len(test_down)} " + (f"down_err={last_down_err[:45]}" if last_down_err else ""),
                    f"score={score:.3f}",
                ],
            )
            frame_idx += 1

            if okh and down_ok == len(test_down) and score < best_score:
                best_score = score
                best_goal = goal_hover

    if best_goal is None:
        print_warn("Nessun goal hover con discesa libera (downtest fallisce ovunque).")
        print_warn("Qui serve cambiare posa 02c oppure scegliere approach dal bordo isola (step base).")
        print_warn(f"Frames: {out_dir}")
        controller.stop()
        return

    okb, erb = move_arm_world(controller, best_goal, speed=1)
    save_step_frame(
        controller,
        out_dir,
        f"{frame_idx:03d}_HOVER_BEST",
        [
            "phase=HOVER_BEST",
            f"best_goal=({best_goal['x']:.2f},{best_goal['y']:.2f},{best_goal['z']:.2f})",
            f"best_score={best_score:.3f}",
            f"ok={okb} err={erb[:60]}",
        ],
    )
    frame_idx += 1

    if not okb:
        print_warn("Non riesco a posizionarmi su HOVER_BEST.")
        controller.stop()
        return

    # =========================================================
    # C) VERTICAL DESC: SOLO Y, XZ fissi
    # =========================================================
    fixed_x = best_goal["x"]
    fixed_z = best_goal["z"]

    y_offsets = [0.28, 0.24, 0.20, 0.16, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.00]

    for si, yo in enumerate(y_offsets, start=1):
        goal = {"x": fixed_x, "y": center["y"] + yo, "z": fixed_z}
        okm, erm = move_arm_world(controller, goal, speed=1)

        hand, r = get_hand(controller)
        d = dist3(hand, center) if hand else None
        in_pick, in_touch, n_pick, n_touch, n_held = is_target_in_range(controller, target_id)

        save_step_frame(
            controller,
            out_dir,
            f"{frame_idx:03d}_DESC_{si:02d}",
            [
                f"phase=VERT_DESC {si}/{len(y_offsets)}",
                f"goal=({goal['x']:.2f},{goal['y']:.2f},{goal['z']:.2f})",
                f"ok={okm} " + (f"err={erm[:60]}" if (not okm and erm) else ""),
                f"hand_dist={d:.3f} r={r}" if d is not None else "hand_dist=N/A",
                f"pickupable={n_pick} touched={n_touch} held={n_held}",
                f"in_pick={in_pick} in_touch={in_touch}",
            ],
        )
        frame_idx += 1

        if not okm:
            print_warn(f"Collisione in VERT_DESC: {erm}")
            break

        if in_pick or in_touch:
            pick = controller.step(action="PickupObject")
            okp = pick.metadata.get("lastActionSuccess", False)
            errp = pick.metadata.get("errorMessage", "")

            save_step_frame(controller, out_dir, f"{frame_idx:03d}_PICK", [f"PICK ok={okp} err={errp[:80]}"])
            frame_idx += 1

            if okp:
                print_ok(f"[ARM] Pickup riuscito ✅  {target_type} ({target_id})")
                print_ok(f"Frames: {out_dir}")
                controller.stop()
                return
            else:
                print_warn(f"Pickup fallito anche se in range: {errp}")

    print_warn("❌ Non preso: collisione prima del contatto utile o mai in range.")
    print_warn(f"Frames: {out_dir}")
    controller.stop()


if __name__ == "__main__":
    main()
